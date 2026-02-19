"""
Systemic vs individual risk signals for CFPB complaints.
Uses narrative phrase signals + risk_type from taxonomy to score product/issue buckets
for regulatory/operational risk (systemic) vs customer-specific (e.g. "interest too high").
Risk mitigation: systemic = process fix / remediation; individual = existing channels.
"""

import re
from pathlib import Path
from typing import Any

import pandas as pd


# Phrases suggesting systemic (process/systems failure, recurring, many affected)
SYSTEMIC_PHRASES = [
    r"\b(multiple|many|several|repeated|again|recurring)\b",
    r"\b(same\s+problem|same\s+issue|same\s+error)\b",
    r"\b(their\s+system|bank\s+error|company\s+error|system\s+error)\b",
    r"\b(incorrectly|wrong\s+amount|wrong\s+balance|wrong\s+information)\b",
    r"\b(not\s+just\s+me|other\s+customers|many\s+people)\b",
    r"\b(process\s+failure|failed\s+to|never\s+fixed)\b",
    r"\b(automated|automatically\s+charged|without\s+notification)\b",
    r"\b(repossessed|repossession)\s+(without|with\s+no)\b",
    r"\b(investigation|investigated)\s+(and\s+)?(wrong|incorrect|nothing)\b",
]

# Phrases suggesting individual (customer preference / single grievance)
INDIVIDUAL_PHRASES = [
    r"\b(interest\s+too\s+high|rate\s+too\s+high|apr\s+too\s+high)\b",
    r"\b(i\s+think|i\s+feel|i\s+believe)\s+",
    r"\b(my\s+rate|my\s+card|my\s+account)\s+(is|was)\s+",
    r"\b(unfair\s+rate|unfair\s+fee)\b",
    r"\b(just\s+want|only\s+asking)\b",
]

SYSTEMIC_PATTERNS = [re.compile(p, re.I) for p in SYSTEMIC_PHRASES]
INDIVIDUAL_PATTERNS = [re.compile(p, re.I) for p in INDIVIDUAL_PHRASES]


def narrative_systemic_score(text: str) -> tuple[float, int, int]:
    """
    Score a narrative: (systemic_score 0-1, n_systemic_hits, n_individual_hits).
    systemic_score = (systemic_hits - individual_hits) normalized; capped 0-1.
    """
    if not text or not isinstance(text, str):
        return 0.0, 0, 0
    t = text.strip()
    sys_hits = sum(1 for pat in SYSTEMIC_PATTERNS if pat.search(t))
    ind_hits = sum(1 for pat in INDIVIDUAL_PATTERNS if pat.search(t))
    raw = max(-1, min(1, (sys_hits - ind_hits) / 3.0))  # scale
    score = (raw + 1) / 2.0  # 0-1
    return round(score, 3), sys_hits, ind_hits


def load_config(project_root: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError:
        return {}
    path = project_root / "config.yaml"
    if not path.is_file():
        return {}
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def load_taxonomy(project_root: Path, taxonomy_path: Path | None = None) -> dict[str, Any]:
    try:
        import yaml
    except ImportError:
        return {}
    if taxonomy_path is None:
        cfg = load_config(project_root)
        paths_cfg = cfg.get("paths", {})
        taxonomy_path = project_root / paths_cfg.get("taxonomy", "data/taxonomy.yaml")
    if not taxonomy_path.is_file():
        return {}
    with open(taxonomy_path, "r") as f:
        return yaml.safe_load(f) or {}


def risk_type_to_systemic_weight(risk_type: str) -> float:
    """Weight for risk_type: operational -> higher systemic tendency."""
    w = {"operational": 0.5, "service": 0.2, "fraud": 0.3, "pricing_dispute": 0.0, "other": 0.1}
    return w.get(risk_type, 0.1)


def compute_product_issue_risk(
    df: pd.DataFrame,
    taxonomy: dict[str, Any],
) -> pd.DataFrame:
    """
    For each (product, issue) or (canonical_product, canonical_issue): volume, distinct_states,
    mean narrative systemic score, risk_type weight, and composite systemic_risk_score 0-1.
    Expects df with product, issue, optional canonical_product, risk_type, consumer_complaint_narrative.
    """
    if df.empty or "product" not in df.columns or "issue" not in df.columns:
        return pd.DataFrame()
    df = df.copy()
    if "risk_type" not in df.columns or "canonical_product" not in df.columns:
        from scripts.standardize_taxonomy import apply_taxonomy as _apply_tax
        df = _apply_tax(df, taxonomy)
    col = "consumer_complaint_narrative"
    if col in df.columns:
        scores = df[col].fillna("").astype(str).apply(narrative_systemic_score)
        df["_sys_score"] = [s[0] for s in scores]
        df["_sys_hits"] = [s[1] for s in scores]
        df["_ind_hits"] = [s[2] for s in scores]
    else:
        df["_sys_score"] = 0.0
        df["_sys_hits"] = 0
        df["_ind_hits"] = 0

    agg = df.groupby(["canonical_product", "canonical_issue"], dropna=False).agg(
        volume=("issue", "size"),
        distinct_states=("state", "nunique") if "state" in df.columns else ("issue", lambda _: 0),
        mean_narrative_systemic=("_sys_score", "mean"),
        risk_type=("risk_type", "first"),
    ).reset_index()
    if "distinct_states" not in agg.columns:
        agg["distinct_states"] = 0
    agg["risk_type_weight"] = agg["risk_type"].apply(risk_type_to_systemic_weight)
    # Composite: 0.35 narrative + 0.35 risk_type + 0.2 geographic spread + 0.1 volume (log norm)
    import numpy as np
    vol_norm = np.log1p(agg["volume"]) / max(np.log1p(agg["volume"].max()), 1)
    state_norm = np.minimum(agg["distinct_states"] / 50.0, 1.0)
    agg["systemic_risk_score"] = (
        0.35 * agg["mean_narrative_systemic"].fillna(0)
        + 0.35 * agg["risk_type_weight"]
        + 0.2 * state_norm
        + 0.1 * vol_norm
    ).clip(0, 1).round(3)
    agg["risk_tier"] = agg["systemic_risk_score"].apply(
        lambda x: "systemic" if x >= 0.5 else ("elevated" if x >= 0.35 else "individual")
    )
    return agg


def main() -> int:
    """Write product-issue risk table to config path (paths.risk_signals)."""
    import sys
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from scripts.standardize_taxonomy import apply_taxonomy as apply_tax

    cfg = load_config(project_root)
    paths_cfg = cfg.get("paths", {})
    taxonomy_path = project_root / paths_cfg.get("taxonomy", "data/taxonomy.yaml")
    taxonomy = load_taxonomy(project_root, taxonomy_path=taxonomy_path)

    complaints_path = project_root / paths_cfg.get("complaints_parquet", "data/wells_fargo_complaints.parquet")
    standardized_path = project_root / paths_cfg.get(
        "complaints_standardized_parquet",
        "data/wells_fargo_complaints_standardized.parquet",
    )
    risk_path = project_root / paths_cfg.get("risk_signals", "data/risk_signals.parquet")

    # Prefer standardized complaints when available for consistent canonical fields.
    input_path = standardized_path if standardized_path.is_file() else complaints_path
    if not input_path.is_file():
        print(f"Complaints not found at {input_path}.", file=sys.stderr)
        return 1
    df = pd.read_parquet(input_path)
    if "canonical_product" not in df.columns or "canonical_issue" not in df.columns or "risk_type" not in df.columns:
        df = apply_tax(df, taxonomy)
    risk_df = compute_product_issue_risk(df, taxonomy)
    risk_path.parent.mkdir(parents=True, exist_ok=True)
    risk_df.to_parquet(risk_path, index=False)
    print(f"Wrote {risk_path} ({len(risk_df)} product-issue rows)")
    return 0


if __name__ == "__main__":
    import sys as _sys
    _sys.exit(main())
