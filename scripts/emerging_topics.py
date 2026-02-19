"""
Compute emerging topics from CFPB Wells Fargo complaints (structured product/issue).
Reads config and Parquet; aggregates by time window; outputs emerging list and optional summary.
Uses pathlib for cross-platform (Mac/Windows). Logs inputs and output path.
"""

import json
import logging
import sys
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import Any

import pandas as pd
import numpy as np


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
    if not cfg:
        raise SystemExit("config.yaml is empty or invalid")
    return cfg


def compute_emerging_products(
    df: pd.DataFrame,
    time_window_days: int,
    prior_window_days: int,
    growth_threshold: float,
    top_n_rank: int,
    top_n_emerging: int,
    reference_date: pd.Timestamp | None = None,
    min_current_volume: int = 10,
    min_prior_volume: int = 5,
    top_issues_per_product: int = 5,
    seasonality_window_days: int = 90,
    persistence_min_runs: int = 1,
    hide_new_until_persistent: bool = False,
    previous_emerging: list[dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    """
    Product-level emergence: aggregate by product (not product×issue) so volumes are meaningful.
    Current/prior windows; growth and rank at product level; only surface products with
    current_volume >= min_current_volume. For each emerging product, attach top_issues (issue + count).
    reference_date: anchor for windows (e.g. end of data).
    """
    product_col = "canonical_product" if "canonical_product" in df.columns else "product"
    issue_col = "canonical_issue" if "canonical_issue" in df.columns else "issue"
    required_cols = {"date_received", product_col, issue_col}
    if df.empty or any(col not in df.columns for col in required_cols):
        return [], pd.DataFrame()
    has_state = "state" in df.columns

    df = df.copy()
    df["date_received"] = pd.to_datetime(df["date_received"], errors="coerce")
    df = df.dropna(subset=["date_received", product_col, issue_col])
    if df.empty:
        return [], pd.DataFrame()

    now = reference_date if reference_date is not None else pd.Timestamp.utcnow()
    if getattr(now, "tzinfo", None) is not None:
        now = now.tz_localize(None)
    current_end = now
    current_start = now - timedelta(days=time_window_days)
    prior_end = current_start
    prior_start = prior_end - timedelta(days=prior_window_days)

    # Baseline window for seasonality (e.g., last 90 days before current window)
    baseline_end = prior_start
    baseline_start = baseline_end - timedelta(days=seasonality_window_days)
    dr = pd.to_datetime(df["date_received"])
    if dr.dt.tz is not None:
        dr = dr.dt.tz_convert("UTC").dt.tz_localize(None)
    df["_dr"] = dr
    df["in_current"] = (df["_dr"] >= current_start) & (df["_dr"] < current_end)
    df["in_prior"] = (df["_dr"] >= prior_start) & (df["_dr"] < prior_end)
    df["in_baseline"] = (df["_dr"] >= baseline_start) & (df["_dr"] < baseline_end)

    # Product-level counts (current and prior); use canonical_product if present
    current_by_product = (
        df.loc[df["in_current"]]
        .groupby(product_col, dropna=False)
        .agg(
            current_volume=(product_col, "size"),
            distinct_states=("state", "nunique") if has_state else (product_col, lambda _: 0),
        )
        .reset_index()
    )
    if not has_state:
        current_by_product["distinct_states"] = 0
    prior_by_product = (
        df.loc[df["in_prior"]]
        .groupby(product_col, dropna=False)
        .size()
        .reset_index(name="prior_volume")
    )

    baseline_by_product = (
        df.loc[df["in_baseline"]]
        .groupby(product_col, dropna=False)
        .size()
        .reset_index(name="baseline_volume")
    )

    merged = (
        current_by_product.merge(prior_by_product, on=product_col, how="outer")
        .merge(baseline_by_product, on=product_col, how="outer")
    ).fillna(0)
    merged["current_volume"] = merged["current_volume"].astype(int)
    merged["prior_volume"] = merged["prior_volume"].astype(int)
    merged["growth_ratio"] = merged.apply(
        lambda r: r["current_volume"] / r["prior_volume"] if r["prior_volume"] > 0 else (float("inf") if r["current_volume"] > 0 else 1.0),
        axis=1,
    )
    merged["growth_pct"] = merged["growth_ratio"].apply(
        lambda x: (x - 1) * 100 if x != float("inf") else None
    )
    merged["rank_current"] = merged["current_volume"].rank(method="min", ascending=False).astype(int)
    merged["rank_prior"] = merged["prior_volume"].rank(method="min", ascending=False).astype(int)
    merged["rank_improved"] = merged["rank_prior"] - merged["rank_current"]

    merged["baseline_volume"] = merged["baseline_volume"].astype(int)
    merged["lift_vs_baseline"] = merged.apply(
        lambda r: r["current_volume"] / r["baseline_volume"] if r["baseline_volume"] > 0 else (float("inf") if r["current_volume"] > 0 else 1.0),
        axis=1,
    )

    growth_meaningful = (merged["growth_ratio"] >= growth_threshold) & (merged["prior_volume"] >= min_prior_volume)
    rank_emerging = (merged["rank_improved"] > 0) & (merged["rank_current"] <= top_n_rank)
    merged["emerging"] = growth_meaningful | rank_emerging
    merged["meets_min_volume"] = merged["current_volume"] >= min_current_volume
    emerging_df = merged.loc[merged["emerging"] & merged["meets_min_volume"]].copy()

    # Persistence filter: keep only items present in previous_emerging when required
    if persistence_min_runs > 1:
        prev_set = {str(item.get("product")) for item in (previous_emerging or [])}
        emerging_df["persistent"] = emerging_df[product_col].astype(str).isin(prev_set)
        if hide_new_until_persistent:
            emerging_df = emerging_df[emerging_df["persistent"]]
    else:
        emerging_df["persistent"] = True

    emerging_df = emerging_df.sort_values(by=["growth_ratio", "current_volume"], ascending=[False, False]).head(top_n_emerging)
    emerging_df["distinct_states"] = emerging_df["distinct_states"].fillna(0).astype(int)

    # Issue-level counts in current window (for drill-down)
    issue_counts = (
        df.loc[df["in_current"]]
        .groupby([product_col, issue_col], dropna=False)
        .size()
        .reset_index(name="current_volume")
    )
    issue_counts = issue_counts.rename(columns={product_col: "_product", issue_col: "_issue"})

    records: list[dict[str, Any]] = []
    for _, row in emerging_df.iterrows():
        product = str(row[product_col])
        takeaway = _actionable_takeaway_product(row)
        g_pct = row.get("growth_pct")
        if pd.isna(g_pct) or row["prior_volume"] == 0 or row["growth_ratio"] == float("inf"):
            g_pct = None
        elif g_pct is not None:
            g_pct = round(float(g_pct), 1)
        g_ratio = row["growth_ratio"]
        product_val = row[product_col]
        top_issues = (
            issue_counts.loc[issue_counts["_product"].astype(str) == str(product_val)]
            .nlargest(top_issues_per_product, "current_volume")
            .apply(lambda r: {"issue": str(r["_issue"]), "current_volume": int(r["current_volume"])}, axis=1)
            .tolist()
        )
        records.append({
            "product": product,
            "rank_current": int(row["rank_current"]),
            "rank_prior": int(row["rank_prior"]),
            "current_volume": int(row["current_volume"]),
            "prior_volume": int(row["prior_volume"]),
            "baseline_volume": int(row.get("baseline_volume", 0)),
            **({"lift_vs_baseline": None} if pd.isna(row.get("lift_vs_baseline")) or row.get("lift_vs_baseline") == float("inf") else {"lift_vs_baseline": float(row.get("lift_vs_baseline", 0))}),
            "distinct_states": int(row.get("distinct_states", 0)),
            "growth_ratio": float(g_ratio) if g_ratio != float("inf") and not pd.isna(g_ratio) else None,
            "growth_pct": g_pct,
            "top_issues": top_issues,
            "takeaway": takeaway,
            "persistent": bool(row.get("persistent", True)),
        })

    summary_df = merged.copy()
    if "product" not in summary_df.columns and product_col in summary_df.columns:
        summary_df["product"] = summary_df[product_col]
    return records, summary_df


def compute_hierarchy_summary(
    df: pd.DataFrame,
    reference_date: pd.Timestamp,
    time_window_days: int,
    prior_window_days: int,
    seasonality_window_days: int,
) -> pd.DataFrame:
    """
    Build hierarchical summary (LOB -> product_line -> issue) with current/prior/baseline volumes,
    growth and rank metrics. Requires date_received and product/issue; uses canonical_* when present.
    """
    if df.empty or "date_received" not in df.columns:
        return pd.DataFrame()
    product_col = "canonical_product" if "canonical_product" in df.columns else "product"
    line_col = "product_line" if "product_line" in df.columns else None
    issue_col = "canonical_issue" if "canonical_issue" in df.columns else "issue"

    now = reference_date
    if getattr(now, "tzinfo", None) is not None:
        now = now.tz_localize(None)
    current_end = now
    current_start = now - timedelta(days=time_window_days)
    prior_end = current_start
    prior_start = prior_end - timedelta(days=prior_window_days)
    baseline_end = prior_start
    baseline_start = baseline_end - timedelta(days=seasonality_window_days)

    dr = pd.to_datetime(df["date_received"], errors="coerce")
    if dr.dt.tz is not None:
        dr = dr.dt.tz_convert("UTC").dt.tz_localize(None)
    df = df.copy()
    df["_dr"] = dr

    def window_flags(frame: pd.DataFrame) -> pd.DataFrame:
        frame["in_current"] = (frame["_dr"] >= current_start) & (frame["_dr"] < current_end)
        frame["in_prior"] = (frame["_dr"] >= prior_start) & (frame["_dr"] < prior_end)
        frame["in_baseline"] = (frame["_dr"] >= baseline_start) & (frame["_dr"] < baseline_end)
        return frame

    df = window_flags(df)

    def agg_level(group_cols: list[str], level_name: str, parent_getter) -> pd.DataFrame:
        cur = df.loc[df["in_current"]].groupby(group_cols, dropna=False).size().reset_index(name="current_volume")
        prior = df.loc[df["in_prior"]].groupby(group_cols, dropna=False).size().reset_index(name="prior_volume")
        base = df.loc[df["in_baseline"]].groupby(group_cols, dropna=False).size().reset_index(name="baseline_volume")
        merged = cur.merge(prior, on=group_cols, how="outer").merge(base, on=group_cols, how="outer").fillna(0)
        merged["current_volume"] = merged["current_volume"].astype(int)
        merged["prior_volume"] = merged["prior_volume"].astype(int)
        merged["baseline_volume"] = merged["baseline_volume"].astype(int)
        merged["growth_ratio"] = merged.apply(
            lambda r: r["current_volume"] / r["prior_volume"] if r["prior_volume"] > 0 else (float("inf") if r["current_volume"] > 0 else 1.0),
            axis=1,
        )
        merged["growth_pct"] = merged["growth_ratio"].apply(lambda x: (x - 1) * 100 if x != float("inf") else None)
        merged["rank_current"] = merged["current_volume"].rank(method="min", ascending=False).astype(int)
        merged["rank_prior"] = merged["prior_volume"].rank(method="min", ascending=False).astype(int)
        merged["rank_improved"] = merged["rank_prior"] - merged["rank_current"]
        merged["level"] = level_name
        merged["name"] = merged[group_cols[-1]].astype(str)
        merged["parent"] = merged.apply(parent_getter, axis=1)
        return merged

    # LOB level (product)
    lob_df = agg_level([product_col], "lob", lambda r: "root")

    # Product line level (optional)
    if line_col and line_col in df.columns:
        line_df = agg_level([product_col, line_col], "product_line", lambda r: str(r[product_col]))
    else:
        line_df = pd.DataFrame()

    # Issue level
    issue_group_cols = [product_col]
    if line_col and line_col in df.columns:
        issue_group_cols.append(line_col)
    issue_group_cols.append(issue_col)
    issue_df = agg_level(issue_group_cols, "issue", lambda r: str(r[line_col]) if line_col and line_col in r else str(r[product_col]))

    hierarchy = pd.concat([lob_df, line_df, issue_df], ignore_index=True, sort=False)
    return hierarchy


def _actionable_takeaway_product(row: pd.Series) -> str:
    """One-line business takeaway for an emerging product (product-level)."""
    product = str(row.get("canonical_product", row.get("product", "")))
    growth_ratio = row.get("growth_ratio")
    rank_current = row.get("rank_current")
    rank_prior = row.get("rank_prior")
    current_volume = int(row.get("current_volume", 0))
    prior_volume = int(row.get("prior_volume", 0))

    parts = [f"{product}"]
    if growth_ratio is not None and prior_volume > 0 and growth_ratio != float("inf"):
        pct = (growth_ratio - 1) * 100
        parts.append(f"complaints up {pct:.0f}% vs prior period ({current_volume} in current window)")
    elif current_volume > 0 and prior_volume == 0:
        parts.append(f"new spike in complaints this period ({current_volume} in current window)")
    if row.get("rank_improved", 0) > 0:
        parts.append(f"now #{int(rank_current)} (was #{int(rank_prior)})")
    parts.append("— consider reviewing process and customer communications.")
    return " ".join(parts)


def _actionable_takeaway(row: pd.Series) -> str:
    """One-line business takeaway for an emerging topic."""
    product = str(row.get("product", ""))
    issue = str(row.get("issue", ""))
    growth_ratio = row.get("growth_ratio")
    rank_current = row.get("rank_current")
    rank_prior = row.get("rank_prior")
    current_volume = int(row.get("current_volume", 0))
    prior_volume = int(row.get("prior_volume", 0))

    parts = [f"{product} – {issue}"]
    if growth_ratio is not None and prior_volume > 0 and growth_ratio != float("inf"):
        pct = (growth_ratio - 1) * 100
        parts.append(f"complaints up {pct:.0f}% vs prior period")
    elif current_volume > 0 and prior_volume == 0:
        parts.append("new spike in complaints this period")
    if row.get("rank_improved", 0) > 0:
        parts.append(f"now #{int(rank_current)} (was #{int(rank_prior)})")
    parts.append("— consider reviewing process and customer communications.")
    return " ".join(parts)


def main() -> int:
    """Read config and Parquet; compute emerging topics; write JSON and optional summary Parquet."""
    logger = setup_logging()
    start_time = datetime.now(timezone.utc)
    stage = "analysis"

    project_root = Path(__file__).resolve().parent.parent
    cfg = load_config(project_root)
    paths_cfg = cfg.get("paths", {})
    emergence_cfg = cfg.get("emergence", {})

    complaints_path = project_root / paths_cfg.get("complaints_parquet", "data/wells_fargo_complaints.parquet")
    standardized_path = project_root / paths_cfg.get("complaints_standardized_parquet", "data/wells_fargo_complaints_standardized.parquet")
    emerging_path = project_root / paths_cfg.get("emerging_topics", "data/emerging_topics.json")
    summary_path = project_root / paths_cfg.get("emerging_summary", "data/emerging_summary.parquet")
    hierarchy_path = project_root / paths_cfg.get("emerging_hierarchy", "data/emerging_hierarchy.parquet")
    hierarchy_json_path = project_root / paths_cfg.get("emerging_hierarchy_json", "data/emerging_hierarchy.json")
    status_path = project_root / paths_cfg.get("pipeline_status", "data/pipeline_status.json")

    time_window_days = int(emergence_cfg.get("time_window_days", 30))
    prior_window_days = int(emergence_cfg.get("prior_window_days", 30))
    growth_threshold = float(emergence_cfg.get("growth_threshold", 1.5))
    top_n_rank = int(emergence_cfg.get("top_n_rank", 5))
    top_n_emerging = int(emergence_cfg.get("top_n_emerging", 10))
    min_current_volume = int(emergence_cfg.get("min_current_volume", 10))
    min_prior_volume = int(emergence_cfg.get("min_prior_volume", 5))
    seasonality_window_days = int(emergence_cfg.get("seasonality_window_days", 90))
    persistence_min_runs = int(emergence_cfg.get("persistence_min_runs", 1))
    hide_new_until_persistent = bool(emergence_cfg.get("hide_new_until_persistent", False))

    if not complaints_path.is_file():
        logger.error("Complaints Parquet not found at %s. Run download_cfpb.py first.", complaints_path)
        return 1

    if standardized_path.is_file():
        logger.info("Reading standardized Parquet from %s", standardized_path)
        df = pd.read_parquet(standardized_path)
    else:
        logger.info("Reading Parquet from %s", complaints_path)
        df = pd.read_parquet(complaints_path)
        if "canonical_product" not in df.columns:
            try:
                from scripts.standardize_taxonomy import load_taxonomy as load_tax, apply_taxonomy as apply_tax
                tax = load_tax(project_root)
                df = apply_tax(df, tax)
                logger.info("Applied taxonomy (canonical_product, risk_type)")
            except Exception as e:
                logger.warning("Taxonomy not applied: %s", e)
    logger.info("Loaded %d rows", len(df))

    # Use end of data as reference so "current" = last N days of data, "prior" = N days before that (not "now")
    df["date_received"] = pd.to_datetime(df["date_received"], errors="coerce")
    df = df.dropna(subset=["date_received"])
    ref = df["date_received"].max()
    if getattr(ref, "tzinfo", None) is not None:
        ref = ref.tz_localize(None)
    logger.info("Reference date for windows: %s (end of data)", ref)

    # Load prior emerging (for persistence) if exists
    previous_emerging: list[dict[str, Any]] | None = None
    if emerging_path.is_file():
        try:
            with open(emerging_path, "r") as f:
                previous_emerging = json.load(f).get("emerging", [])
        except Exception:
            previous_emerging = None

    records, summary_df = compute_emerging_products(
        df,
        time_window_days=time_window_days,
        prior_window_days=prior_window_days,
        growth_threshold=growth_threshold,
        top_n_rank=top_n_rank,
        top_n_emerging=top_n_emerging,
        reference_date=ref,
        min_current_volume=min_current_volume,
        min_prior_volume=min_prior_volume,
        seasonality_window_days=seasonality_window_days,
        persistence_min_runs=persistence_min_runs,
        hide_new_until_persistent=hide_new_until_persistent,
        previous_emerging=previous_emerging,
    )

    emerging_path.parent.mkdir(parents=True, exist_ok=True)
    with open(emerging_path, "w") as f:
        json.dump({"emerging": records, "computed_at": datetime.now(timezone.utc).isoformat()}, f, indent=2)
    logger.info("Wrote emerging topics to %s (%d items)", emerging_path, len(records))

    if not summary_df.empty:
        summary_df.to_parquet(summary_path, index=False)
        logger.info("Wrote summary to %s", summary_path)

    # Hierarchical summary for dashboard (LOB -> product_line -> issue)
    hierarchy_df = compute_hierarchy_summary(
        df,
        reference_date=ref,
        time_window_days=time_window_days,
        prior_window_days=prior_window_days,
        seasonality_window_days=seasonality_window_days,
    )
    if not hierarchy_df.empty:
        hierarchy_path.parent.mkdir(parents=True, exist_ok=True)
        hierarchy_df.to_parquet(hierarchy_path, index=False)
        try:
            hierarchy_records = hierarchy_df.to_dict(orient="records")
            with open(hierarchy_json_path, "w") as f:
                json.dump({"hierarchy": hierarchy_records, "computed_at": datetime.now(timezone.utc).isoformat()}, f, indent=2)
        except Exception:
            logger.warning("Could not write hierarchy JSON", exc_info=True)
        logger.info("Wrote hierarchy summary to %s", hierarchy_path)

    duration = (datetime.now(timezone.utc) - start_time).total_seconds()
    if status_path.is_file():
        try:
            with open(status_path, "r") as f:
                status = json.load(f)
        except Exception:
            status = {}
    else:
        status = {}
    try:
        status.update({
            "last_run_iso": datetime.now(timezone.utc).isoformat(),
            "status": "success",
            "duration_seconds": round(duration, 2),
            "stage": stage,
        })
        with open(status_path, "w") as f:
            json.dump(status, f, indent=2)
    except Exception:
        logger.warning("Could not write pipeline_status", exc_info=True)
    logger.info("Stage %s completed in %.2f s", stage, duration)
    return 0


if __name__ == "__main__":
    sys.exit(main())
