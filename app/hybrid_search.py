"""
Hybrid search: keyword (product/issue) + semantic similarity over corpus embeddings.
Loads complaints and narrative_embeddings (complaint_idx, embedding); filters by canonical
product/issue; optionally ranks by cosine similarity to a query embedding.
"""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SEARCH_TEXT_COLUMNS = [
    "consumer_complaint_narrative",
    "issue",
    "product",
    "canonical_issue",
    "canonical_product",
    "sub_issue",
    "sub_product",
]


def load_embeddings_table(embeddings_path: Path) -> pd.DataFrame:
    """Load narrative_embeddings.parquet (complaint_idx, embedding as list)."""
    if not embeddings_path.is_file():
        return pd.DataFrame()
    df = pd.read_parquet(embeddings_path)
    if not {"complaint_idx", "embedding"}.issubset(df.columns):
        return pd.DataFrame()
    return df


def _ensure_complaint_idx_column(df: pd.DataFrame) -> pd.DataFrame:
    """Guarantee an integer complaint_idx column for stable embedding joins."""
    out = df.copy()
    if "complaint_idx" not in out.columns:
        out["complaint_idx"] = out.index
    out["complaint_idx"] = pd.to_numeric(out["complaint_idx"], errors="coerce")
    out = out.dropna(subset=["complaint_idx"])
    if out.empty:
        return out
    out["complaint_idx"] = out["complaint_idx"].astype(int)
    return out


def _keyword_mask(df: pd.DataFrame, query: str) -> pd.Series:
    """Build OR mask across complaint text columns."""
    mask_parts = []
    for col in SEARCH_TEXT_COLUMNS:
        if col in df.columns:
            mask_parts.append(df[col].astype(str).str.contains(query, case=False, na=False))
    if not mask_parts:
        return pd.Series(False, index=df.index)
    mask = mask_parts[0]
    for m in mask_parts[1:]:
        mask = mask | m
    return mask


def embed_query(model: Any, query: str) -> np.ndarray:
    """Single query -> (dim,) normalized vector."""
    if not query or not model:
        return np.array([])
    try:
        arr = model.encode([query.strip()], normalize_embeddings=True)
        return arr[0] if len(arr) else np.array([])
    except Exception:
        # Gracefully degrade if model/torch/numpy stack is unavailable
        return np.array([])


def hybrid_search(
    complaints_df: pd.DataFrame,
    embeddings_df: pd.DataFrame,
    query_text: str | None = None,
    canonical_product: str | None = None,
    canonical_issue: str | None = None,
    product_raw: str | None = None,
    issue_raw: str | None = None,
    top_k: int = 20,
    model: Any = None,
    use_canonical: bool = True,
) -> pd.DataFrame:
    """
    Filter complaints by product/issue (keyword), then optionally rank by semantic
    similarity to query_text. Returns top_k rows with column similarity_score if query given.
    If embeddings/model are unavailable, falls back to keyword contains search.
    """
    if complaints_df.empty:
        return pd.DataFrame()
    df = _ensure_complaint_idx_column(complaints_df)
    # Apply canonical filter: use canonical_product/canonical_issue if present
    if use_canonical and "canonical_product" in df.columns and canonical_product:
        df = df[df["canonical_product"].astype(str) == str(canonical_product)]
    elif product_raw:
        df = df[df["product"].astype(str) == str(product_raw)]
    if use_canonical and "canonical_issue" in df.columns and canonical_issue:
        df = df[df["canonical_issue"].astype(str) == str(canonical_issue)]
    elif issue_raw:
        df = df[df["issue"].astype(str) == str(issue_raw)]
    if df.empty:
        return pd.DataFrame()

    has_query = bool(query_text and str(query_text).strip())
    q = str(query_text).strip() if has_query else ""
    embeddings_ready = (
        not embeddings_df.empty
        and {"complaint_idx", "embedding"}.issubset(embeddings_df.columns)
    )
    can_semantic = has_query and embeddings_ready and model is not None

    # If semantic is unavailable, use keyword-only fallback for explicit query.
    if has_query and not can_semantic:
        mask = _keyword_mask(df, q)
        if mask.any():
            return df.loc[mask].head(top_k)
        return pd.DataFrame()

    # If we have embeddings and a model, attempt semantic
    if embeddings_ready:
        emb = embeddings_df.copy()
        emb["complaint_idx"] = pd.to_numeric(emb["complaint_idx"], errors="coerce")
        emb = emb.dropna(subset=["complaint_idx"])
        emb["complaint_idx"] = emb["complaint_idx"].astype(int)
        emb = emb.drop_duplicates(subset=["complaint_idx"], keep="last").set_index("complaint_idx")

        in_emb = df["complaint_idx"].astype(int).isin(emb.index)
        if not in_emb.any():
            if has_query:
                mask = _keyword_mask(df, q)
                return df.loc[mask].head(top_k) if mask.any() else pd.DataFrame()
            return df.head(top_k).assign(similarity_score=np.nan)
        df_emb = df.loc[in_emb].copy()
        rows_to_keep: list[int] = []
        vec_list: list[np.ndarray] = []
        dim: int | None = None
        for row_idx, cid in zip(df_emb.index.tolist(), df_emb["complaint_idx"].astype(int).tolist()):
            vec = np.asarray(emb.loc[cid, "embedding"], dtype=float)
            if vec.ndim != 1 or vec.size == 0:
                continue
            if dim is None:
                dim = int(vec.shape[0])
            if int(vec.shape[0]) != dim:
                continue
            rows_to_keep.append(row_idx)
            vec_list.append(vec)

        if not vec_list:
            if has_query:
                mask = _keyword_mask(df_emb, q)
                return df_emb.loc[mask].head(top_k) if mask.any() else pd.DataFrame()
            return df_emb.head(top_k).assign(similarity_score=np.nan)
        df_emb = df_emb.loc[rows_to_keep].copy()
        vecs = np.vstack(vec_list)
        if has_query and model is not None and vecs.size:
            qv = embed_query(model, q)
            if qv.size == vecs.shape[1]:
                scores = np.dot(vecs, qv)
                df_emb = df_emb.assign(similarity_score=scores)
                df_emb = df_emb.sort_values("similarity_score", ascending=False).head(top_k)
            else:
                mask = _keyword_mask(df_emb, q)
                if mask.any():
                    df_emb = df_emb.loc[mask].head(top_k)
                else:
                    return pd.DataFrame()
        else:
            df_emb = df_emb.head(top_k)
        return df_emb

    # Fallback: keyword contains search over complaint text fields.
    if has_query:
        mask = _keyword_mask(df, q)
        if mask.any():
            return df.loc[mask].head(top_k)
        return pd.DataFrame()
    # Default: return top_k (keyword-only)
    return df.head(top_k)


def get_search_canonical_options(
    complaints_df: pd.DataFrame,
    taxonomy: dict[str, Any],
) -> tuple[list[str], list[str]]:
    """Return sorted unique canonical_product and canonical_issue for dropdowns."""
    from scripts.standardize_taxonomy import apply_taxonomy
    if complaints_df.empty:
        return [], []
    df = apply_taxonomy(complaints_df, taxonomy) if "canonical_product" not in complaints_df.columns else complaints_df
    products = sorted(df["canonical_product"].dropna().astype(str).unique().tolist()) if "canonical_product" in df.columns else []
    issues = sorted(df["canonical_issue"].dropna().astype(str).unique().tolist()) if "canonical_issue" in df.columns else []
    return products, issues
