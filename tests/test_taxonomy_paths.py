"""
Tests for config-driven taxonomy path resolution.
"""

import sys
from pathlib import Path

# Project root = parent of tests/; make scripts importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.risk_signals import load_taxonomy as load_risk_taxonomy
from scripts.standardize_taxonomy import load_taxonomy as load_standardize_taxonomy


def test_standardize_load_taxonomy_uses_config_path(tmp_path: Path) -> None:
    cfg_path = tmp_path / "config.yaml"
    tax_path = tmp_path / "custom" / "taxonomy_custom.yaml"
    tax_path.parent.mkdir(parents=True, exist_ok=True)
    tax_path.write_text("canonical_products:\n  Mortgage: Mortgage\n", encoding="utf-8")
    cfg_path.write_text("paths:\n  taxonomy: custom/taxonomy_custom.yaml\n", encoding="utf-8")

    tax = load_standardize_taxonomy(tmp_path)
    assert tax.get("canonical_products", {}).get("Mortgage") == "Mortgage"


def test_risk_load_taxonomy_uses_config_path(tmp_path: Path) -> None:
    cfg_path = tmp_path / "config.yaml"
    tax_path = tmp_path / "alt" / "taxonomy.yaml"
    tax_path.parent.mkdir(parents=True, exist_ok=True)
    tax_path.write_text("issue_risk_type:\n  Fees or interest: pricing_dispute\n", encoding="utf-8")
    cfg_path.write_text("paths:\n  taxonomy: alt/taxonomy.yaml\n", encoding="utf-8")

    tax = load_risk_taxonomy(tmp_path)
    assert tax.get("issue_risk_type", {}).get("Fees or interest") == "pricing_dispute"
