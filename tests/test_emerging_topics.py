"""
Unit tests for emerging-topics logic.
Given a small fixture (synthetic counts by product/issue and period), assert that the correct
items are flagged as emerging (growth and/or rank) and that thresholds are applied as in config.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

# Project root = parent of tests/; make scripts importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.emerging_topics import compute_emerging_products


def test_emerging_topics_empty_df() -> None:
    """Empty DataFrame returns empty list and empty summary."""
    records, summary = compute_emerging_products(
        pd.DataFrame(),
        time_window_days=30,
        prior_window_days=30,
        growth_threshold=1.5,
        top_n_rank=5,
        top_n_emerging=10,
    )
    assert records == []
    assert summary.empty


def test_emerging_topics_missing_columns() -> None:
    """DataFrame missing date_received or product or issue returns empty."""
    df = pd.DataFrame({"product": ["A"], "issue": ["X"]})
    records, summary = compute_emerging_products(
        df,
        time_window_days=30,
        prior_window_days=30,
        growth_threshold=1.5,
        top_n_rank=5,
        top_n_emerging=10,
    )
    assert records == []
    assert summary.empty


def test_emerging_topics_growth_threshold() -> None:
    """Product with growth >= threshold and min volume is flagged as emerging."""
    ref = pd.Timestamp("2024-06-01", tz="UTC")
    if ref.tzinfo is not None:
        ref = ref.tz_localize(None)
    current_start = ref - pd.Timedelta(days=30)
    prior_start = current_start - pd.Timedelta(days=30)
    # Current: 10, Prior: 6 for Mortgage -> ratio 10/6 >= 1.5; min_current_volume=10, min_prior_volume=5
    df = pd.DataFrame({
        "date_received": (
            [current_start + pd.Timedelta(days=i) for i in range(10)]
            + [prior_start + pd.Timedelta(days=i) for i in range(6)]
        ),
        "product": ["Mortgage"] * 16,
        "issue": ["Loan modification"] * 16,
    })
    records, summary = compute_emerging_products(
        df,
        time_window_days=30,
        prior_window_days=30,
        growth_threshold=1.5,
        top_n_rank=5,
        top_n_emerging=10,
        reference_date=ref,
        min_current_volume=10,
        min_prior_volume=5,
    )
    assert len(records) >= 1
    products = [r["product"] for r in records]
    assert "Mortgage" in products
    growth_ratios = [r["growth_ratio"] for r in records if r.get("growth_ratio") is not None]
    assert any(g >= 1.5 for g in growth_ratios)
    assert all("top_issues" in r for r in records)


def test_emerging_topics_respects_top_n_emerging() -> None:
    """At most top_n_emerging products returned."""
    ref = pd.Timestamp("2024-06-01", tz="UTC")
    if ref.tzinfo is not None:
        ref = ref.tz_localize(None)
    current_start = ref - pd.Timedelta(days=30)
    prior_start = current_start - pd.Timedelta(days=30)
    # 5 products, each with 10 current and 6 prior (growth meaningful, min volume met)
    rows = []
    for i in range(5):
        rows.extend([
            {"date_received": current_start + pd.Timedelta(days=j), "product": f"Product_{i}", "issue": f"Issue_{i}"}
            for j in range(10)
        ])
        rows.extend([
            {"date_received": prior_start + pd.Timedelta(days=j), "product": f"Product_{i}", "issue": f"Issue_{i}"}
            for j in range(6)
        ])
    df = pd.DataFrame(rows)
    records, _ = compute_emerging_products(
        df,
        time_window_days=30,
        prior_window_days=30,
        growth_threshold=1.5,
        top_n_rank=5,
        top_n_emerging=3,
        reference_date=ref,
        min_current_volume=10,
        min_prior_volume=5,
    )
    assert len(records) <= 3


def test_emerging_topics_with_canonical_only_columns() -> None:
    """Canonical-only inputs (no raw product/issue) still compute emerging records."""
    ref = pd.Timestamp("2024-06-01", tz="UTC")
    if ref.tzinfo is not None:
        ref = ref.tz_localize(None)
    current_start = ref - pd.Timedelta(days=30)
    prior_start = current_start - pd.Timedelta(days=30)
    df = pd.DataFrame({
        "date_received": (
            [current_start + pd.Timedelta(days=i) for i in range(10)]
            + [prior_start + pd.Timedelta(days=i) for i in range(6)]
        ),
        "canonical_product": ["Mortgage"] * 16,
        "canonical_issue": ["Loan modification"] * 16,
    })
    records, summary = compute_emerging_products(
        df,
        time_window_days=30,
        prior_window_days=30,
        growth_threshold=1.5,
        top_n_rank=5,
        top_n_emerging=10,
        reference_date=ref,
        min_current_volume=10,
        min_prior_volume=5,
    )
    assert records
    assert records[0]["product"] == "Mortgage"
    assert "product" in summary.columns


def test_emerging_topics_summary_has_expected_columns() -> None:
    """Summary DataFrame has product-level growth_ratio, rank_current, prior_volume, etc."""
    ref = pd.Timestamp("2024-06-01", tz="UTC")
    if ref.tzinfo is not None:
        ref = ref.tz_localize(None)
    current_start = ref - pd.Timedelta(days=30)
    df = pd.DataFrame({
        "date_received": [current_start],
        "product": ["Checking"],
        "issue": ["Overdraft"],
    })
    _, summary = compute_emerging_products(
        df,
        time_window_days=30,
        prior_window_days=30,
        growth_threshold=1.5,
        top_n_rank=5,
        top_n_emerging=10,
        reference_date=ref,
    )
    assert not summary.empty
    for col in ["product", "current_volume", "prior_volume", "growth_ratio", "rank_current", "rank_prior", "emerging"]:
        assert col in summary.columns
