"""
Download CFPB consumer complaints, filter by company (e.g. Wells Fargo), and persist as Parquet.
Writes manifest and pipeline_status for lineage and operator visibility.
Uses pathlib for cross-platform (Mac/Windows). Exit 1 on failure.
"""

import json
import logging
import sys
import zipfile
from pathlib import Path
from datetime import datetime, timezone

import requests
import pandas as pd

# Expected columns for schema validation (CFPB CSV may use "Date received" etc.; we normalize)
REQUIRED_COLUMNS = {
    "date_received",
    "company",
    "product",
    "issue",
}

# Map common CSV header variants to canonical names
COLUMN_ALIASES = {
    "date received": "date_received",
    "date_received": "date_received",
    "company": "company",
    "product": "product",
    "sub_product": "sub_product",
    "sub-product": "sub_product",
    "issue": "issue",
    "sub_issue": "sub_issue",
    "sub-issue": "sub_issue",
    "consumer_complaint_narrative": "consumer_complaint_narrative",
    "consumer complaint narrative": "consumer_complaint_narrative",
    "state": "state",
    "tags": "tags",
}


def setup_logging() -> logging.Logger:
    """Configure structured logging with level and timestamps."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger(__name__)


def load_config_yaml(config_path: Path) -> dict:
    """Load config from config.yaml with proper YAML parsing."""
    try:
        import yaml
    except ImportError:
        raise SystemExit("PyYAML required for config. Install: pip install pyyaml")
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    if not cfg:
        raise SystemExit("config.yaml is empty or invalid")
    return cfg


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names to snake_case canonical set for CFPB CSV."""
    rename = {}
    for c in df.columns:
        key = str(c).strip().lower().replace(" ", "_").replace("-", "_")
        canonical = COLUMN_ALIASES.get(key, key)
        rename[c] = canonical
    return df.rename(columns=rename)


def main() -> int:
    """Download CFPB zip, filter by company, save Parquet, write manifest and pipeline status."""
    logger = setup_logging()
    start_time = datetime.now(timezone.utc)
    stage = "download"

    # Resolve paths relative to project root (parent of scripts/)
    project_root = Path(__file__).resolve().parent.parent
    config_path = project_root / "config.yaml"
    if not config_path.is_file():
        logger.error("config.yaml not found at %s", config_path)
        return 1

    cfg = load_config_yaml(config_path)
    cfpb_cfg = cfg.get("cfpb", {})
    paths_cfg = cfg.get("paths", {})

    source_url = cfpb_cfg.get("source_url", "https://files.consumerfinance.gov/ccdb/complaints.csv.zip")
    company_pattern = cfpb_cfg.get("company_filter_pattern", "Wells Fargo")
    dedup_subset = cfpb_cfg.get("dedup_subset") or []
    min_filtered_rows = int(cfpb_cfg.get("min_filtered_rows", 0))

    data_dir = project_root / paths_cfg.get("data_dir", "data")
    complaints_path = project_root / paths_cfg.get("complaints_parquet", "data/wells_fargo_complaints.parquet")
    manifest_path = project_root / paths_cfg.get("manifest", "data/manifest.json")
    status_path = project_root / paths_cfg.get("pipeline_status", "data/pipeline_status.json")

    data_dir.mkdir(parents=True, exist_ok=True)

    def write_pipeline_status(status: str, duration_seconds: float) -> None:
        payload = {
            "last_run_iso": datetime.now(timezone.utc).isoformat(),
            "status": status,
            "duration_seconds": round(duration_seconds, 2),
            "stage": stage,
        }
        with open(status_path, "w") as f:
            json.dump(payload, f, indent=2)

    try:
        logger.info("Downloading from %s", source_url)
        resp = requests.get(source_url, timeout=120)
        resp.raise_for_status()
        zip_path = data_dir / "complaints.csv.zip"
        zip_path.write_bytes(resp.content)
        logger.info("Downloaded %s bytes", len(resp.content))

        with zipfile.ZipFile(zip_path, "r") as z:
            names = z.namelist()
            csv_name = next((n for n in names if n.endswith(".csv")), None)
            if not csv_name:
                logger.error("No CSV file in zip")
                write_pipeline_status("failure", (datetime.now(timezone.utc) - start_time).total_seconds())
                return 1
            with z.open(csv_name) as f:
                df = pd.read_csv(f, low_memory=False)

        row_count_full = len(df)
        logger.info("Loaded %d rows from CSV", row_count_full)

        df = normalize_columns(df)
        missing = REQUIRED_COLUMNS - set(df.columns)
        if missing:
            logger.error("Missing required columns: %s. Found: %s", missing, list(df.columns))
            write_pipeline_status("failure", (datetime.now(timezone.utc) - start_time).total_seconds())
            return 1

        if df.empty:
            logger.error("DataFrame is empty after load")
            write_pipeline_status("failure", (datetime.now(timezone.utc) - start_time).total_seconds())
            return 1

        # Parse date_received
        if "date_received" in df.columns:
            df["date_received"] = pd.to_datetime(df["date_received"], errors="coerce")

        # Filter by company
        company_col = df["company"].astype(str)
        mask = company_col.str.contains(company_pattern, case=False, na=False)
        df_filtered = df.loc[mask].copy()

        # Drop exact/near duplicates using configured subset if available
        if dedup_subset:
            cols_present = [c for c in dedup_subset if c in df_filtered.columns]
            if cols_present:
                before = len(df_filtered)
                df_filtered = df_filtered.drop_duplicates(subset=cols_present)
                logger.info("Deduped %d -> %d rows using %s", before, len(df_filtered), cols_present)
            else:
                logger.info("Dedup subset configured but columns missing; skipping dedup: %s", dedup_subset)
        row_count_filtered = len(df_filtered)
        logger.info("Filtered to %d rows for company pattern '%s'", row_count_filtered, company_pattern)

        if min_filtered_rows and row_count_filtered < min_filtered_rows:
            logger.warning("Filtered rows %d below min_filtered_rows %d", row_count_filtered, min_filtered_rows)

        if df_filtered.empty:
            logger.warning("No rows after company filter; writing empty Parquet and manifest anyway")

        complaints_path.parent.mkdir(parents=True, exist_ok=True)
        df_filtered.to_parquet(complaints_path, index=False)
        logger.info("Wrote Parquet to %s", complaints_path)

        last_refresh_iso = datetime.now(timezone.utc).isoformat()
        manifest = {
            "source_url": source_url,
            "last_refresh_iso": last_refresh_iso,
            "row_count_full": row_count_full,
            "row_count_filtered": row_count_filtered,
        }
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)
        logger.info("Wrote manifest to %s", manifest_path)

        duration = (datetime.now(timezone.utc) - start_time).total_seconds()
        write_pipeline_status("success", duration)
        logger.info("Pipeline stage %s completed in %.2f s", stage, duration)
        return 0

    except requests.RequestException as e:
        logger.exception("Network error: %s", e)
        write_pipeline_status("failure", (datetime.now(timezone.utc) - start_time).total_seconds())
        return 1
    except Exception as e:
        logger.exception("Unexpected error: %s", e)
        write_pipeline_status("failure", (datetime.now(timezone.utc) - start_time).total_seconds())
        return 1


if __name__ == "__main__":
    sys.exit(main())
