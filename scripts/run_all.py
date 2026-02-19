"""
Run the full CFPB Wells Fargo pipeline in order:
1. download_cfpb.py
2. standardize_taxonomy.py
3. risk_signals.py
4. emerging_topics.py
5. narrative_topics.py (optional; set RUN_NARRATIVE=1 or --narrative to enable)

Uses subprocess so each step runs in isolation. Exit code 0 only if all enabled steps succeed.
"""

import argparse
import subprocess
import sys
from pathlib import Path


def run_script(name: str, script_path: Path, project_root: Path) -> int:
    """Run a script; return its exit code."""
    cmd = [sys.executable, str(script_path)]
    result = subprocess.run(cmd, cwd=str(project_root))
    return result.returncode


def main() -> int:
    parser = argparse.ArgumentParser(description="Run full CFPB Wells Fargo pipeline")
    parser.add_argument(
        "--narrative",
        action="store_true",
        help="Also run narrative_topics.py (embeddings + BERTopic; slow)",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip download_cfpb.py (use existing Parquet)",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    scripts_dir = project_root / "scripts"

    steps = [
        ("download_cfpb", "download_cfpb.py", not args.skip_download),
        ("standardize_taxonomy", "standardize_taxonomy.py", True),
        ("risk_signals", "risk_signals.py", True),
        ("emerging_topics", "emerging_topics.py", True),
        ("narrative_topics", "narrative_topics.py", args.narrative),
    ]

    for label, filename, enabled in steps:
        if not enabled:
            print(f"[SKIP] {label}")
            continue
        script_path = scripts_dir / filename
        if not script_path.is_file():
            print(f"[ERROR] {script_path} not found", file=sys.stderr)
            return 1
        print(f"[RUN] {label} ...")
        code = run_script(label, script_path, project_root)
        if code != 0:
            print(f"[FAIL] {label} exited with {code}", file=sys.stderr)
            return code
        print(f"[OK] {label}")

    print("Pipeline finished successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
