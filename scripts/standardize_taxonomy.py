"""
Apply CFPB taxonomy for customer redress: map raw product/issue to redress LOB,
product_line (sub-LOB refinement), canonical issue, and risk_type.
LOB = who owns the complaint (Consumer Banking, Credit Card, Mortgage, etc.);
money transfer is a product line under Consumer Banking, not its own LOB.
Reads taxonomy path from config (paths.taxonomy); use apply_taxonomy(df, taxonomy)
or run as script to write standardized Parquet.
"""

import sys
from pathlib import Path
from typing import Any

import pandas as pd


def load_config(project_root: Path) -> dict[str, Any]:
    """Load config.yaml from project root."""
    try:
        import yaml
    except ImportError:
        raise SystemExit("PyYAML required. pip install pyyaml")
    config_path = project_root / "config.yaml"
    if not config_path.is_file():
        return {}
    with open(config_path, "r") as f:
        return yaml.safe_load(f) or {}


def load_taxonomy(project_root: Path, taxonomy_path: Path | None = None) -> dict[str, Any]:
    """Load taxonomy YAML from config path (or explicit path override)."""
    if taxonomy_path is None:
        cfg = load_config(project_root)
        paths = cfg.get("paths", {})
        taxonomy_path = project_root / paths.get("taxonomy", "data/taxonomy.yaml")
    path = taxonomy_path
    if not path.is_file():
        raise SystemExit(f"Taxonomy not found: {path}")
    with open(path, "r") as f:
        import yaml
        return yaml.safe_load(f) or {}


def apply_taxonomy(df: pd.DataFrame, taxonomy: dict[str, Any]) -> pd.DataFrame:
    """
    Add canonical_product (redress LOB), product_line (refinement), canonical_issue, risk_type.
    Preserves raw product/issue. Unmapped product -> LOB "Other", product_line "Other".
    """
    if df.empty:
        return df
    out = df.copy()
    product_map = taxonomy.get("canonical_products") or {}
    product_line_map = taxonomy.get("product_line") or {}
    issue_risk = taxonomy.get("issue_risk_type") or {}
    issue_canon = taxonomy.get("canonical_issues") or {}
    default_risk = issue_risk.get("_default", "other")
    default_line = product_line_map.get("_default", "Other")

    def map_lob(raw: Any) -> str:
        s = str(raw).strip() if pd.notna(raw) else ""
        return product_map.get(s, "Other")

    def map_product_line(raw: Any) -> str:
        s = str(raw).strip() if pd.notna(raw) else ""
        return product_line_map.get(s, default_line)

    def map_risk(raw: Any) -> str:
        s = str(raw).strip() if pd.notna(raw) else ""
        return issue_risk.get(s, default_risk)

    def map_issue_label(raw: Any) -> str:
        s = str(raw).strip() if pd.notna(raw) else ""
        v = issue_canon.get(s)
        return str(v) if v is not None else s

    if "product" in out.columns:
        out["canonical_product"] = out["product"].apply(map_lob)
        out["product_line"] = out["product"].apply(map_product_line)
    if "issue" in out.columns:
        out["risk_type"] = out["issue"].apply(map_risk)
        out["canonical_issue"] = out["issue"].apply(map_issue_label)
    return out


def main() -> int:
    project_root = Path(__file__).resolve().parent.parent
    cfg = load_config(project_root)
    paths = cfg.get("paths", {})
    taxonomy_path = project_root / paths.get("taxonomy", "data/taxonomy.yaml")
    taxonomy = load_taxonomy(project_root, taxonomy_path=taxonomy_path)
    complaints_path = project_root / paths.get("complaints_parquet", "data/wells_fargo_complaints.parquet")
    out_path = project_root / paths.get("complaints_standardized_parquet", "data/wells_fargo_complaints_standardized.parquet")

    if not complaints_path.is_file():
        print("Complaints Parquet not found. Run download_cfpb.py first.", file=sys.stderr)
        return 1

    df = pd.read_parquet(complaints_path)
    df = apply_taxonomy(df, taxonomy)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    print(f"Wrote standardized Parquet to {out_path} ({len(df)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
