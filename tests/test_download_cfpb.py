"""
Unit tests for download + filter logic.
Test with a small cached CSV slice or mocked response; assert Wells Fargo filter and column presence.
"""

import sys
import zipfile
from pathlib import Path
from io import BytesIO

import pandas as pd
import pytest

# Project root = parent of tests/; make scripts importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def test_normalize_columns() -> None:
    """Normalize_columns maps CFPB-style headers to canonical snake_case."""
    import sys
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from scripts.download_cfpb import normalize_columns

    df = pd.DataFrame({
        "Date received": ["2024-01-01"],
        "Company": ["Wells Fargo Bank"],
        "Product": ["Mortgage"],
        "Issue": ["Loan modification"],
    })
    out = normalize_columns(df)
    assert "date_received" in out.columns
    assert "company" in out.columns
    assert "product" in out.columns
    assert "issue" in out.columns


def test_wells_fargo_filter() -> None:
    """Filter keeps rows where company contains 'Wells Fargo' (case-insensitive)."""
    df = pd.DataFrame({
        "company": ["Wells Fargo Bank", "WELLS FARGO COMPANY", "Other Bank", "Wells Fargo"],
        "product": ["A", "B", "C", "D"],
        "issue": ["X", "Y", "Z", "W"],
    })
    pattern = "Wells Fargo"
    mask = df["company"].astype(str).str.contains(pattern, case=False, na=False)
    filtered = df.loc[mask]
    assert len(filtered) == 3
    assert "Other Bank" not in filtered["company"].tolist()


def test_required_columns_after_normalize() -> None:
    """After normalize, required columns (date_received, company, product, issue) are present."""
    import sys
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from scripts.download_cfpb import normalize_columns, REQUIRED_COLUMNS

    df = pd.DataFrame({
        "Date received": ["2024-01-01"],
        "Company": ["Wells Fargo"],
        "Product": ["Mortgage"],
        "Issue": ["Loan modification"],
    })
    out = normalize_columns(df)
    missing = REQUIRED_COLUMNS - set(out.columns)
    assert len(missing) == 0, f"Missing columns: {missing}"


def test_download_mock_zip_parsing() -> None:
    """Parse a minimal in-memory zip with CSV; assert schema and filter apply."""
    csv_content = b"Date received,Company,Product,Issue\n2024-01-01,Wells Fargo Bank,Mortgage,Loan mod\n2024-01-02,Other Bank,Checking,Fee\n"
    buf = BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("complaints.csv", csv_content)
    buf.seek(0)
    with zipfile.ZipFile(buf, "r") as z:
        names = z.namelist()
        csv_name = next((n for n in names if n.endswith(".csv")), None)
        assert csv_name is not None
        with z.open(csv_name) as f:
            df = pd.read_csv(f)
    assert "Date received" in df.columns or "date_received" in df.columns
    assert len(df) == 2
    import sys
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from scripts.download_cfpb import normalize_columns
    df = normalize_columns(df)
    mask = df["company"].astype(str).str.contains("Wells Fargo", case=False, na=False)
    filtered = df.loc[mask]
    assert len(filtered) == 1
    assert filtered.iloc[0]["product"] == "Mortgage"
