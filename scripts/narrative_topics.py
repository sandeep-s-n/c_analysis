"""
Narrative pipeline: embed full complaint corpus for hybrid semantic search, store embeddings
in Parquet, run BERTopic on narrative-bearing rows for topic/cluster id, aggregate by time,
apply same emergence logic as structured. Output narrative_emerging.json for dashboard.
Uses pathlib; model id and batch size from config. No vector DB; no custom training.
"""

import json
import logging
import sys
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import Any

import numpy as np
import pandas as pd

# Optional: BERTopic and sentence-transformers (phase 2 deps)
try:
    from sentence_transformers import SentenceTransformer
    import torch
except ImportError:
    SentenceTransformer = None
    torch = None
try:
    from bertopic import BERTopic
except ImportError:
    BERTopic = None


SEMANTIC_FALLBACK_FIELDS = [
    "canonical_product",
    "product",
    "product_line",
    "canonical_issue",
    "issue",
    "sub_product",
    "sub_issue",
    "company_response_to_consumer",
]


def setup_logging() -> logging.Logger:
    """Configure structured logging with level and timestamps."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger(__name__)


def load_config(project_root: Path) -> dict:
    """Load config.yaml from project root."""
    try:
        import yaml
    except ImportError:
        raise SystemExit("PyYAML required. Install: pip install pyyaml")
    config_path = project_root / "config.yaml"
    if not config_path.is_file():
        raise SystemExit("config.yaml not found")
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg or {}


def load_complaints_for_embeddings(complaints_path: Path) -> pd.DataFrame:
    """Load complaints Parquet and build semantic text for every complaint row."""
    df = pd.read_parquet(complaints_path)
    if df.empty:
        return pd.DataFrame()
    df = df.copy()
    df["complaint_idx"] = df.index.astype(int).values
    if "date_received" not in df.columns:
        return pd.DataFrame()
    df["date_received"] = pd.to_datetime(df["date_received"], errors="coerce")
    df = df.dropna(subset=["date_received"])

    narrative = (
        df.get("consumer_complaint_narrative", pd.Series("", index=df.index))
        .fillna("")
        .astype(str)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )
    df["has_narrative"] = narrative != ""

    fallback_cols = [c for c in SEMANTIC_FALLBACK_FIELDS if c in df.columns]
    if fallback_cols:
        fallback_df = df[fallback_cols].fillna("").astype(str)
        for col in fallback_cols:
            fallback_df[col] = fallback_df[col].str.replace(r"\s+", " ", regex=True).str.strip()
        fallback = fallback_df.apply(
            lambda row: " | ".join([v for v in row.tolist() if v]),
            axis=1,
        ).str.strip(" |")
    else:
        fallback = pd.Series("", index=df.index)

    semantic_text = narrative.where(df["has_narrative"], fallback)
    semantic_text = semantic_text.fillna("").astype(str).str.strip()
    semantic_text = semantic_text.where(
        semantic_text != "",
        "Complaint with sparse structured details and no narrative text provided.",
    )
    df["semantic_text"] = semantic_text
    return df.reset_index(drop=True)


def embed_batch(
    model: Any,
    texts: list[str],
    batch_size: int,
) -> np.ndarray:
    """Embed texts in batches to avoid OOM. Returns (n, dim) float array."""
    if not texts:
        return np.array([]).reshape(0, 0)
    all_embeds = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        emb = model.encode(batch, normalize_embeddings=True, show_progress_bar=False)
        all_embeds.append(emb)
    return np.vstack(all_embeds)


def compute_emerging_narrative_topics(
    topic_counts_current: pd.DataFrame,
    topic_counts_prior: pd.DataFrame,
    topic_info: dict[int, str],
    growth_threshold: float,
    top_n_rank: int,
    top_n_emerging: int,
    min_current_volume: int = 10,
) -> list[dict[str, Any]]:
    """
    Apply same emergence logic as structured: growth >= threshold or rank improved into top_n.
    Only surface topics with current_volume >= min_current_volume (avoids 2–3 complaint noise).
    topic_counts_* have columns Topic, count (or Frequency). topic_info maps topic_id -> label.
    """
    merged = topic_counts_current.merge(
        topic_counts_prior,
        on="Topic",
        how="outer",
        suffixes=("_current", "_prior"),
    ).fillna(0)
    count_cur = "count_current" if "count_current" in merged.columns else "Frequency_current"
    count_prior = "count_prior" if "count_prior" in merged.columns else "Frequency_prior"
    if count_cur not in merged.columns:
        cand = [c for c in merged.columns if "current" in c.lower()]
        count_cur = cand[0] if cand else "count"
    if count_prior not in merged.columns:
        cand = [c for c in merged.columns if "prior" in c.lower()]
        count_prior = cand[0] if cand else "count"
    merged["current_volume"] = merged[count_cur].astype(int)
    merged["prior_volume"] = merged[count_prior].astype(int)
    merged["growth_ratio"] = merged.apply(
        lambda r: r["current_volume"] / r["prior_volume"] if r["prior_volume"] > 0 else (float("inf") if r["current_volume"] > 0 else 1.0),
        axis=1,
    )
    merged["rank_current"] = merged["current_volume"].rank(method="min", ascending=False).astype(int)
    merged["rank_prior"] = merged["prior_volume"].rank(method="min", ascending=False).astype(int)
    merged["rank_improved"] = merged["rank_prior"] - merged["rank_current"]
    merged["emerging"] = (
        (merged["growth_ratio"] >= growth_threshold)
        | ((merged["rank_improved"] > 0) & (merged["rank_current"] <= top_n_rank))
    )
    merged["meets_min_volume"] = merged["current_volume"] >= min_current_volume
    emerging_df = merged.loc[merged["emerging"] & merged["meets_min_volume"]].sort_values(
        by=["growth_ratio", "current_volume"],
        ascending=[False, False],
    ).head(top_n_emerging)

    records: list[dict[str, Any]] = []
    for _, row in emerging_df.iterrows():
        topic_id = int(row["Topic"])
        label = topic_info.get(topic_id, f"Topic {topic_id}")
        cur = int(row["current_volume"])
        prior = int(row["prior_volume"])
        growth = row["growth_ratio"]
        takeaway = f"{label} — complaints up {(growth - 1) * 100:.0f}% vs prior period" if prior > 0 and growth != float("inf") else f"{label} — new spike in narrative volume."
        takeaway += " Consider reviewing process and customer communications."
        records.append({
            "topic_id": topic_id,
            "topic_label": label,
            "current_volume": cur,
            "prior_volume": prior,
            "growth_ratio": float(growth) if growth != float("inf") else None,
            "takeaway": takeaway,
        })
    return records


def main() -> int:
    """Run narrative pipeline: load narratives, embed, store Parquet, BERTopic, emergence, write JSON."""
    logger = setup_logging()
    start_time = datetime.now(timezone.utc)
    project_root = Path(__file__).resolve().parent.parent
    cfg = load_config(project_root)
    paths_cfg = cfg.get("paths", {})
    emergence_cfg = cfg.get("emergence", {})

    complaints_path = project_root / paths_cfg.get("complaints_parquet", "data/wells_fargo_complaints.parquet")
    embeddings_path = project_root / paths_cfg.get("narrative_embeddings", "data/narrative_embeddings.parquet")
    emerging_path = project_root / paths_cfg.get("narrative_emerging", "data/narrative_emerging.json")
    topics_table_path = project_root / paths_cfg.get("narrative_topics_table", "data/narrative_topics.parquet")
    status_path = project_root / paths_cfg.get("pipeline_status", "data/pipeline_status.json")

    embedding_model_id = cfg.get("embedding_model_id", "sentence-transformers/all-MiniLM-L6-v2")
    embedding_device = cfg.get("embedding_device", "auto")
    batch_size = int(cfg.get("narrative_embed_batch_size", 64))
    time_window_days = int(emergence_cfg.get("time_window_days", 30))
    prior_window_days = int(emergence_cfg.get("prior_window_days", 30))
    growth_threshold = float(emergence_cfg.get("growth_threshold", 1.5))
    top_n_rank = int(emergence_cfg.get("top_n_rank", 5))
    top_n_emerging = int(emergence_cfg.get("top_n_emerging", 10))
    min_current_volume = int(emergence_cfg.get("min_current_volume", 10))

    if not complaints_path.is_file():
        logger.error("Complaints Parquet not found. Run download_cfpb.py first.")
        return 1

    if SentenceTransformer is None or torch is None:
        logger.error(
            "Narrative pipeline requires torch and sentence-transformers. "
            "Install: pip install torch sentence-transformers"
        )
        return 1

    df_all = load_complaints_for_embeddings(complaints_path)
    if df_all.empty:
        logger.warning("No valid complaints rows found for embeddings. Skipping narrative pipeline.")
        return 0

    n_total = len(df_all)
    n_narrative = int(df_all["has_narrative"].sum()) if "has_narrative" in df_all.columns else 0
    logger.info("Loaded %d complaints for embeddings (%d with narratives)", n_total, n_narrative)

    # Embed with Hugging Face model (configurable)
    logger.info("Loading embedding model: %s (device=%s)", embedding_model_id, embedding_device)
    device_kwargs = {}
    if torch is None:
        device_kwargs["device"] = "cpu"
    elif embedding_device and embedding_device != "auto":
        device_kwargs["device"] = embedding_device
    model = SentenceTransformer(embedding_model_id, **device_kwargs)
    docs_all = df_all["semantic_text"].astype(str).tolist()
    logger.info("Embedding full complaint corpus in batches of %d", batch_size)
    embeddings_all = embed_batch(model, docs_all, batch_size)

    # Store embeddings in Parquet (complaint_idx joins back to complaints for hybrid search)
    out_df = pd.DataFrame({
        "complaint_idx": df_all["complaint_idx"].values,
        "date_received": df_all["date_received"].values,
        "has_narrative": df_all["has_narrative"].values,
    })
    out_df["embedding"] = [emb.tolist() for emb in embeddings_all]
    embeddings_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(embeddings_path, index=False)
    logger.info("Wrote corpus embeddings to %s", embeddings_path)

    records: list[dict[str, Any]] = []
    topics_out = pd.DataFrame(columns=["complaint_idx", "topic_id", "date_received"])

    # Narrative clustering remains narrative-only; hybrid semantic search now uses full-corpus embeddings.
    if BERTopic is None:
        logger.warning("BERTopic not installed. Skipping narrative topic clustering; hybrid semantic search remains available.")
    elif n_narrative <= 0:
        logger.warning("No rows with non-null consumer_complaint_narrative. Skipping narrative topic clustering.")
    else:
        df = df_all[df_all["has_narrative"]].copy()
        docs = df["consumer_complaint_narrative"].astype(str).str.strip().tolist()
        narrative_embeddings = embeddings_all[df_all["has_narrative"].to_numpy()]

        logger.info("Running BERTopic on %d narrative rows", len(df))
        topic_model = BERTopic(embedding_model=embedding_model_id, verbose=False)
        topics, _ = topic_model.fit_transform(docs, embeddings=narrative_embeddings)
        df["topic_id"] = topics

        # Persist topic assignments for drill-through (complaint_idx, topic_id, date_received, canonical fields, narrative)
        topics_out = df[[
            "complaint_idx",
            "topic_id",
            "date_received",
        ]].copy()
        for col in ["canonical_product", "canonical_issue", "product", "issue", "product_line", "state", "consumer_complaint_narrative"]:
            if col in df.columns:
                topics_out[col] = df[col]

        # Topic labels from BERTopic (top words per topic)
        topic_info: dict[int, str] = {}
        if hasattr(topic_model, "get_topic") and topic_model.get_topic is not None:
            for tid in set(t for t in topics if t is not None and t >= 0):
                tinfo = topic_model.get_topic(tid)
                if tinfo:
                    label = ", ".join(w for w, _ in tinfo[:3])
                    topic_info[tid] = label
                else:
                    topic_info[tid] = f"Topic {tid}"
        else:
            topic_info = {t: f"Topic {t}" for t in set(topics) if t is not None and t >= 0}

        # Time buckets: use end of data as reference (same as structured) so "current" = last N days of data
        ref = df["date_received"].max()
        if getattr(ref, "tzinfo", None) is not None:
            ref = ref.tz_localize(None)
        current_end = ref
        current_start = ref - timedelta(days=time_window_days)
        prior_end = current_start
        prior_start = prior_end - timedelta(days=prior_window_days)
        dr = pd.to_datetime(df["date_received"])
        if dr.dt.tz is not None:
            dr = dr.dt.tz_convert("UTC").dt.tz_localize(None)
        df["_dr"] = dr
        in_current = (df["_dr"] >= current_start) & (df["_dr"] < current_end)
        in_prior = (df["_dr"] >= prior_start) & (df["_dr"] < prior_end)

        topic_counts_current = df.loc[in_current & (df["topic_id"] >= 0)].groupby("topic_id", dropna=False).size().reset_index(name="count")
        topic_counts_current = topic_counts_current.rename(columns={"topic_id": "Topic"})
        topic_counts_prior = df.loc[in_prior & (df["topic_id"] >= 0)].groupby("topic_id", dropna=False).size().reset_index(name="count")
        topic_counts_prior = topic_counts_prior.rename(columns={"topic_id": "Topic"})

        records = compute_emerging_narrative_topics(
            topic_counts_current,
            topic_counts_prior,
            topic_info,
            growth_threshold,
            top_n_rank,
            top_n_emerging,
            min_current_volume=min_current_volume,
        )

    topics_table_path.parent.mkdir(parents=True, exist_ok=True)
    topics_out.to_parquet(topics_table_path, index=False)
    logger.info("Wrote topic assignments to %s (%d rows)", topics_table_path, len(topics_out))

    emerging_path.parent.mkdir(parents=True, exist_ok=True)
    with open(emerging_path, "w") as f:
        json.dump({
            "emerging": records,
            "computed_at": datetime.now(timezone.utc).isoformat(),
        }, f, indent=2)
    logger.info("Wrote narrative emerging topics to %s (%d items)", emerging_path, len(records))

    duration = (datetime.now(timezone.utc) - start_time).total_seconds()
    if status_path.is_file():
        try:
            with open(status_path, "r") as f:
                status = json.load(f)
            status["last_run_iso"] = datetime.now(timezone.utc).isoformat()
            status["status"] = "success"
            status["duration_seconds"] = round(duration, 2)
            status["stage"] = "narratives"
            with open(status_path, "w") as f:
                json.dump(status, f, indent=2)
        except Exception:
            pass
    logger.info("Narrative pipeline completed in %.2f s", duration)
    return 0


if __name__ == "__main__":
    sys.exit(main())
