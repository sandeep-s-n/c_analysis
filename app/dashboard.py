"""
CFPB Wells Fargo Consumer Complaints — executive dashboard.
Tufte/NYT editorial standard: high data-ink ratio, serif hierarchy, muted palette,
direct labeling, generous white space. Uses pathlib for cross-platform (Mac/Windows).
"""

from pathlib import Path
from typing import Any
import subprocess
import sys

import pandas as pd
import plotly.express as px
import streamlit as st

# Reuse pipeline logic for on-the-fly emerging/hierarchy tied to user-selected dates
from scripts.emerging_topics import compute_emerging_products, compute_hierarchy_summary

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Wells Fargo brand (red #D41C2C, yellow #FCCC44, white) + Tufte clarity
# Ref: wells fargo brand palette; wells fargo jobs / talent community styling
WF_RED = "#D41C2C"
WF_YELLOW = "#FCCC44"
WF_WHITE = "#FFFFFF"

TUFTE_CSS = f"""
<style>
/* Layout + background: NYT/Tufte quiet canvas */
.block-container {{ padding-top: 1.25rem; padding-bottom: 2rem; max-width: 960px; background: #f7f7f3; }}
.main .block-container {{ max-width: 980px; }}

/* Header strip: slim Wells Fargo red with restrained type */
.wf-header {{ background: {WF_RED}; color: {WF_WHITE}; padding: 0.35rem 0.9rem; margin: -1rem -1rem 1rem -1rem; font-family: 'Georgia', 'Times New Roman', serif; font-weight: 600; font-size: 0.95rem; letter-spacing: 0.01em; }}
.wf-header a {{ color: {WF_WHITE}; text-decoration: none; }}
.wf-accent {{ height: 2px; background: linear-gradient(90deg, {WF_RED} 0%, {WF_YELLOW} 100%); margin-bottom: 1rem; opacity: 0.6; }}

/* Typography: refined serif heads, neutral body */
h1, h2, h3 {{ font-family: 'Georgia', 'Times New Roman', serif !important; font-weight: 600; color: #1a1a1a; }}
h1 {{ font-size: 2rem; letter-spacing: -0.01em; margin-bottom: 0.35rem; }}
h2 {{ font-size: 1.2rem; letter-spacing: 0.01em; margin-top: 1.75rem; margin-bottom: 0.6rem; border-bottom: 1px solid #d7d2cb; padding-bottom: 0.25rem; }}
h3 {{ font-size: 1.03rem; margin-top: 1.1rem; margin-bottom: 0.4rem; }}
p, li, .stMarkdown {{ font-family: 'Source Sans Pro', 'Helvetica Neue', Arial, sans-serif; color: #2b2b2b; }}

/* Metrics row: subtle cards, minimal red */
[data-testid="stMetric"] {{ background: #ffffff; border: 1px solid #e6e1d9; box-shadow: 0 3px 10px rgba(0,0,0,0.03); padding: 0.6rem 0.75rem; border-radius: 4px; }}
[data-testid="stMetric"] label {{ font-family: 'Source Sans Pro','Helvetica Neue',Arial,sans-serif; font-size: 0.72rem; color: #5c5c5c; text-transform: uppercase; letter-spacing: 0.05em; }}
[data-testid="stMetric"] [data-testid="stMetricValue"] {{ font-family: 'Georgia', serif; font-size: 1.65rem; color: #b0171f; }}

/* Tables: fine rules, light headers */
[data-testid="stDataFrame"] {{ font-size: 0.92rem; color: #1f1f1f; }}
[data-testid="stDataFrame"] table {{ border: 1px solid #e3dfd7 !important; background: #fdfdfb; }}
[data-testid="stDataFrame"] th {{ background: #f3efe8 !important; font-weight: 600; color: #2a2a2a; border-bottom: 1px solid #e0dbd3; }}
[data-testid="stDataFrame"] td {{ border-color: #e9e4dc !important; }}

/* Sidebar: muted */
[data-testid="stSidebar"] {{ background: #f4f1ea; }}
[data-testid="stSidebar"] .stMarkdown {{ font-size: 0.92rem; color: #3a3a3a; }}
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {{ color: #8c111a; }}

/* Captions and small text */
.caption-tufte {{ font-size: 0.8rem; color: #666; margin-top: 0.2rem; margin-bottom: 0.45rem; line-height: 1.45; }}

/* Section dividers */
hr {{ border: none; border-top: 1px solid #d7d2cb; margin: 1.8rem 0 1rem 0; }}

/* Expanders: understated labels */
[data-testid="stExpander"] summary {{ font-weight: 500; color: #1f1f1f; }}

/* Buttons: outlined, low chroma */
.stButton > button {{ border-radius: 3px; border: 1px solid #b0171f; background: #ffffff; color: #b0171f; padding: 0.4rem 0.9rem; }}
.stButton > button:hover {{ background: #b0171f; color: #ffffff; border-color: #b0171f; }}

/* Charts: keep backgrounds clean */
canvas {{ background: transparent !important; }}
</style>
"""


def load_config() -> dict:
    try:
        import yaml
    except ImportError:
        return {}
    config_path = PROJECT_ROOT / "config.yaml"
    if not config_path.is_file():
        return {}
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg or {}


def load_manifest() -> dict[str, Any]:
    paths_cfg = load_config().get("paths", {})
    manifest_path = PROJECT_ROOT / paths_cfg.get("manifest", "data/manifest.json")
    if not manifest_path.is_file():
        return {}
    import json
    with open(manifest_path, "r") as f:
        return json.load(f)


def load_emerging_topics() -> list[dict[str, Any]]:
    paths_cfg = load_config().get("paths", {})
    emerging_path = PROJECT_ROOT / paths_cfg.get("emerging_topics", "data/emerging_topics.json")
    if not emerging_path.is_file():
        return []
    import json
    with open(emerging_path, "r") as f:
        data = json.load(f)
    return data.get("emerging", [])


def load_narrative_emerging() -> list[dict[str, Any]]:
    paths_cfg = load_config().get("paths", {})
    path = PROJECT_ROOT / paths_cfg.get("narrative_emerging", "data/narrative_emerging.json")
    if not path.is_file():
        return []
    import json
    with open(path, "r") as f:
        data = json.load(f)
    return data.get("emerging", [])


def load_narrative_topics_table() -> pd.DataFrame:
    paths_cfg = load_config().get("paths", {})
    path = PROJECT_ROOT / paths_cfg.get("narrative_topics_table", "data/narrative_topics.parquet")
    return pd.read_parquet(path) if path.is_file() else pd.DataFrame()


def load_complaints_df(standardized_first: bool = True) -> pd.DataFrame:
    paths_cfg = load_config().get("paths", {})
    complaints_path = PROJECT_ROOT / paths_cfg.get("complaints_parquet", "data/wells_fargo_complaints.parquet")
    standardized_path = PROJECT_ROOT / paths_cfg.get("complaints_standardized_parquet", "data/wells_fargo_complaints_standardized.parquet")
    if standardized_first and standardized_path.is_file():
        return pd.read_parquet(standardized_path)
    if not complaints_path.is_file():
        return pd.DataFrame()
    df = pd.read_parquet(complaints_path)
    if "canonical_product" not in df.columns:
        try:
            from scripts.standardize_taxonomy import load_taxonomy, apply_taxonomy
            tax = load_taxonomy(PROJECT_ROOT)
            df = apply_taxonomy(df, tax)
        except Exception:
            pass
    return df


def load_hierarchy_summary() -> pd.DataFrame:
    paths_cfg = load_config().get("paths", {})
    path = PROJECT_ROOT / paths_cfg.get("emerging_hierarchy", "data/emerging_hierarchy.parquet")
    return pd.read_parquet(path) if path.is_file() else pd.DataFrame()


def load_risk_signals() -> pd.DataFrame:
    paths_cfg = load_config().get("paths", {})
    path = PROJECT_ROOT / paths_cfg.get("risk_signals", "data/risk_signals.parquet")
    return pd.read_parquet(path) if path.is_file() else pd.DataFrame()


def load_pipeline_status() -> dict[str, Any]:
    paths_cfg = load_config().get("paths", {})
    path = PROJECT_ROOT / paths_cfg.get("pipeline_status", "data/pipeline_status.json")
    if not path.is_file():
        return {}
    import json
    with open(path, "r") as f:
        return json.load(f)


def _max_date(series: pd.Series) -> pd.Timestamp | None:
    """Return max parsed date (naive), or None."""
    s = pd.to_datetime(series, errors="coerce")
    if getattr(s.dt, "tz", None) is not None:
        s = s.dt.tz_convert("UTC").dt.tz_localize(None)
    m = s.max()
    return None if pd.isna(m) else m


def _run_narrative_pipeline(timeout_seconds: int = 1800) -> tuple[bool, str]:
    """Run narrative_topics.py and return (ok, tail_logs)."""
    script = PROJECT_ROOT / "scripts" / "narrative_topics.py"
    if not script.is_file():
        return False, f"Script not found: {script}"
    try:
        result = subprocess.run(
            [sys.executable, str(script)],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except Exception as e:
        return False, str(e)
    stdout_tail = "\n".join((result.stdout or "").strip().splitlines()[-5:])
    stderr_tail = "\n".join((result.stderr or "").strip().splitlines()[-5:])
    tail = "\n".join([x for x in [stdout_tail, stderr_tail] if x]).strip()
    return result.returncode == 0, tail


def _recompute_narrative_emerging_for_sidebar(
    narrative_topics_tbl: pd.DataFrame,
    df_indexed: pd.DataFrame,
    product_filter: str,
    start_d,
    end_d,
    window_days: int,
    emergence_cfg: dict[str, Any],
    topic_labels: dict[int, str],
) -> list[dict[str, Any]]:
    """
    Recompute narrative emerging topics for the current sidebar scope (date + optional LOB).
    """
    from scripts.narrative_topics import compute_emerging_narrative_topics

    if narrative_topics_tbl.empty or df_indexed.empty:
        return []
    if not {"topic_id", "complaint_idx"}.issubset(narrative_topics_tbl.columns):
        return []

    lob_col = "canonical_product" if "canonical_product" in df_indexed.columns else "product"
    if lob_col not in df_indexed.columns or "date_received" not in df_indexed.columns:
        return []

    work = narrative_topics_tbl[["topic_id", "complaint_idx"]].copy()
    work = work.dropna(subset=["topic_id", "complaint_idx"])
    if work.empty:
        return []
    work["topic_id"] = work["topic_id"].astype(int)
    work = work[work["topic_id"] >= 0]
    if work.empty:
        return []

    work = work.merge(
        df_indexed[["complaint_idx", "date_received", lob_col]],
        on="complaint_idx",
        how="inner",
    )
    work["date_received"] = pd.to_datetime(work["date_received"], errors="coerce")
    work = work.dropna(subset=["date_received"])
    if work.empty:
        return []

    if product_filter != "All lines of business":
        work = work[work[lob_col].astype(str) == str(product_filter)]
    if work.empty:
        return []

    current_end = pd.Timestamp(end_d) + pd.Timedelta(days=1)
    current_start = current_end - pd.Timedelta(days=window_days)
    prior_end = current_start
    prior_start = prior_end - pd.Timedelta(days=window_days)

    cur = work[(work["date_received"] >= current_start) & (work["date_received"] < current_end)]
    prior = work[(work["date_received"] >= prior_start) & (work["date_received"] < prior_end)]

    topic_counts_current = cur.groupby("topic_id", dropna=False).size().reset_index(name="count").rename(columns={"topic_id": "Topic"})
    topic_counts_prior = prior.groupby("topic_id", dropna=False).size().reset_index(name="count").rename(columns={"topic_id": "Topic"})

    return compute_emerging_narrative_topics(
        topic_counts_current=topic_counts_current,
        topic_counts_prior=topic_counts_prior,
        topic_info=topic_labels,
        growth_threshold=float(emergence_cfg.get("growth_threshold", 1.5)),
        top_n_rank=int(emergence_cfg.get("top_n_rank", 5)),
        top_n_emerging=int(emergence_cfg.get("top_n_emerging", 10)),
        min_current_volume=int(emergence_cfg.get("min_current_volume", 10)),
    )


def _build_topic_label_map(
    narrative_emerging: list[dict[str, Any]],
    narrative_topics_tbl: pd.DataFrame,
) -> dict[int, str]:
    """
    Build topic_id -> label map.
    Priority:
    1) Labels already produced by narrative_emerging.json
    2) Inferred labels from topic assignments table (most common issue/product per topic)
    """
    labels: dict[int, str] = {}
    for item in narrative_emerging:
        try:
            tid = int(item.get("topic_id"))
        except Exception:
            continue
        label = str(item.get("topic_label", "")).strip()
        if label:
            labels[tid] = label

    if narrative_topics_tbl.empty or "topic_id" not in narrative_topics_tbl.columns:
        return labels

    tbl = narrative_topics_tbl.copy()
    tbl = tbl.dropna(subset=["topic_id"])
    if tbl.empty:
        return labels
    tbl["topic_id"] = tbl["topic_id"].astype(int)
    tbl = tbl[tbl["topic_id"] >= 0]
    if tbl.empty:
        return labels

    issue_col = "canonical_issue" if "canonical_issue" in tbl.columns else ("issue" if "issue" in tbl.columns else None)
    product_col = "canonical_product" if "canonical_product" in tbl.columns else ("product" if "product" in tbl.columns else None)

    issue_top: dict[int, str] = {}
    product_top: dict[int, str] = {}

    # Taxonomy maps to canonical display labels when topic table only has raw CFPB values.
    issue_map: dict[str, str] = {}
    product_map: dict[str, str] = {}
    try:
        cfg = load_config()
        paths_cfg = cfg.get("paths", {})
        tax_path = PROJECT_ROOT / paths_cfg.get("taxonomy", "data/taxonomy.yaml")
        if tax_path.is_file():
            import yaml
            tax = yaml.safe_load(tax_path.read_text()) or {}
            issue_map = {
                str(k): str(v)
                for k, v in (tax.get("canonical_issues") or {}).items()
                if v is not None and str(k) != "_default"
            }
            product_map = {
                str(k): str(v)
                for k, v in (tax.get("canonical_products") or {}).items()
                if v is not None
            }
    except Exception:
        issue_map = {}
        product_map = {}

    if issue_col is not None:
        t_issue = tbl[["topic_id", issue_col]].copy()
        t_issue[issue_col] = t_issue[issue_col].fillna("").astype(str).str.strip()
        t_issue = t_issue[t_issue[issue_col] != ""]
        if not t_issue.empty:
            issue_top = (
                t_issue.groupby("topic_id")[issue_col]
                .agg(lambda s: s.value_counts().index[0])
                .to_dict()
            )

    if product_col is not None:
        t_prod = tbl[["topic_id", product_col]].copy()
        t_prod[product_col] = t_prod[product_col].fillna("").astype(str).str.strip()
        t_prod = t_prod[t_prod[product_col] != ""]
        if not t_prod.empty:
            product_top = (
                t_prod.groupby("topic_id")[product_col]
                .agg(lambda s: s.value_counts().index[0])
                .to_dict()
            )

    inferred_labels: dict[int, str] = {}
    for tid in sorted(tbl["topic_id"].unique().tolist()):
        if tid in labels:
            continue
        issue = issue_top.get(tid, "")
        product = product_top.get(tid, "")
        issue = issue_map.get(issue, issue)
        product = product_map.get(product, product)
        if issue and product:
            inferred_labels[tid] = f"{issue} — {product}"
        elif issue:
            inferred_labels[tid] = issue
        elif product:
            inferred_labels[tid] = product

    # If inferred labels repeat across topics, append topic id to disambiguate.
    value_counts: dict[str, int] = {}
    for v in inferred_labels.values():
        value_counts[v] = value_counts.get(v, 0) + 1
    for tid, v in inferred_labels.items():
        if value_counts.get(v, 0) > 1:
            labels[tid] = f"{v} (Topic {tid})"
        else:
            labels[tid] = v

    return labels


COMPLAINT_EXPORT_BASE_COLS = [
    "complaint_idx",
    "complaint_id",
    "date_received",
    "company",
    "state",
    "zip_code",
    "submitted_via",
    "timely_response?",
    "consumer_disputed?",
    "company_response_to_consumer",
    "company_public_response",
    # raw CFPB fields
    "product",
    "sub_product",
    "issue",
    "sub_issue",
    # enhanced taxonomy fields
    "canonical_product",
    "product_line",
    "canonical_issue",
    "risk_type",
    # narrative
    "consumer_complaint_narrative",
]


def _prepare_complaint_export(df: pd.DataFrame, leading_cols: list[str] | None = None) -> pd.DataFrame:
    """Return complaint-level export with stable column order and normalized dates."""
    if df.empty:
        return df.copy()
    out = df.copy()
    if "date_received" in out.columns:
        out["date_received"] = pd.to_datetime(out["date_received"], errors="coerce").dt.strftime("%Y-%m-%d")
    ordered_cols: list[str] = []
    for c in (leading_cols or []) + COMPLAINT_EXPORT_BASE_COLS:
        if c in out.columns and c not in ordered_cols:
            ordered_cols.append(c)
    ordered_cols.extend([c for c in out.columns if c not in ordered_cols])
    return out[ordered_cols]


def main() -> None:
    st.set_page_config(
        page_title="Wells Fargo · Consumer Complaints",
        page_icon="",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(TUFTE_CSS, unsafe_allow_html=True)

    # Wells Fargo branding: header strip (talent community style)
    st.markdown(
        '<div class="wf-header">Wells Fargo · Consumer Complaints Analytics</div>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="wf-accent"></div>', unsafe_allow_html=True)

    cfg = load_config()
    manifest = load_manifest()
    pipeline_status = load_pipeline_status()
    last_refresh = manifest.get("last_refresh_iso", pipeline_status.get("last_run_iso", "—"))
    if isinstance(last_refresh, str) and "T" in last_refresh:
        try:
            from datetime import datetime
            dt = datetime.fromisoformat(last_refresh.replace("Z", "+00:00"))
            data_as_of = dt.strftime("%Y-%m-%d %H:%M UTC")
        except Exception:
            data_as_of = last_refresh
    else:
        data_as_of = str(last_refresh)

    df_full = load_complaints_df(standardized_first=True)
    df = df_full.copy()
    if df.empty:
        st.warning("No complaint data. Run `scripts/download_cfpb.py` from the project root.")
        return

    df["date_received"] = pd.to_datetime(df["date_received"], errors="coerce")
    df = df.dropna(subset=["date_received"])

    # ——— Sidebar: filters and lineage (Tufte: minimal chrome) ———
    with st.sidebar:
        st.markdown("### Filters")
        date_min = df["date_received"].min()
        date_max = df["date_received"].max()
        d_min = date_min.date() if hasattr(date_min, "date") else pd.Timestamp(date_min).date()
        d_max = date_max.date() if hasattr(date_max, "date") else pd.Timestamp(date_max).date()
        from datetime import timedelta
        default_end = d_max
        default_start = (pd.Timestamp(default_end) - timedelta(days=14)).date()
        date_range = st.date_input(
            "Date range",
            value=(default_start, default_end),
            min_value=d_min,
            max_value=d_max,
        )
        lob_col = "canonical_product" if "canonical_product" in df.columns else "product"
        products = ["All lines of business"] + sorted(df[lob_col].dropna().astype(str).unique().tolist())
        product_filter = st.selectbox("Line of business", products)
        st.markdown("---")
        st.markdown(f"<p class='caption-tufte'>Data as of {data_as_of}<br>Source: CFPB Consumer Complaint Database</p>", unsafe_allow_html=True)
        with st.expander("Methods"):
            st.caption("Structured emerging: 30d vs prior 30d; growth ≥1.5× or rank rise. Narrative: sentence-transformers + BERTopic; same emergence logic.")

    # Apply filters
    # Apply date filter; keep window info
    if hasattr(date_range, "__len__") and len(date_range) == 2:
        start_d, end_d = date_range[0], date_range[1]
    else:
        start_d = default_start
        end_d = default_end
    df = df[(df["date_received"].dt.date >= start_d) & (df["date_received"].dt.date <= end_d)]
    window_days = max(1, (pd.Timestamp(end_d) - pd.Timestamp(start_d)).days + 1)
    prior_start = pd.Timestamp(start_d) - pd.Timedelta(days=window_days)
    prior_end = pd.Timestamp(start_d)
    coverage_note = ""
    if prior_start.date() < d_min:
        coverage_note = "Prior window partially outside data range; comparison may be understated."
    if product_filter != "All lines of business":
        lob_col = "canonical_product" if "canonical_product" in df.columns else "product"
        df = df[df[lob_col].astype(str) == product_filter]

    total_complaints = len(df)

    # ——— Headline + dateline + KPI row ———
    st.markdown("# Consumer Complaints")
    st.markdown(f"<p class='caption-tufte'>CFPB complaint data · Wells Fargo · Data as of {data_as_of}</p>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.metric("Complaints reported", f"{total_complaints:,}")
    with k2:
        n_lob = df["product"].nunique() if "product" in df.columns else 0
        st.metric("Lines of business", str(n_lob))
    with k3:
        n_regions = df["state"].nunique() if "state" in df.columns else 0
        st.metric("Regions (states)", str(n_regions))
    with k4:
        n_issues = df["issue"].nunique() if "issue" in df.columns else 0
        st.metric("Distinct issues", str(n_issues))

    if pipeline_status:
        st.caption(
            f"Last pipeline run: {pipeline_status.get('last_run_iso', '—')} · Stage: {pipeline_status.get('stage', '—')} · Status: {pipeline_status.get('status', '—')}"
        )

    st.markdown("---")

    # ——— Lines of business (redress LOB: who owns the complaint) ———
    st.markdown("## Lines of business")
    st.markdown("<p class='caption-tufte'>Redress LOBs: who owns the complaint. Money transfer is a product line under Consumer Banking, not its own LOB.</p>", unsafe_allow_html=True)
    lob_col = "canonical_product" if "canonical_product" in df.columns else "product"
    if lob_col in df.columns and not df.empty:
        lob = df[lob_col].value_counts().reset_index()
        lob.columns = ["Line of business", "Complaints"]
        lob["% of total"] = (lob["Complaints"] / lob["Complaints"].sum() * 100).round(1).astype(str) + "%"
        st.dataframe(lob, use_container_width=True, hide_index=True)
        st.bar_chart(lob.set_index("Line of business")["Complaints"])
        if "product_line" in df.columns:
            pl = df.groupby([lob_col, "product_line"], dropna=False).size().reset_index(name="Complaints")
            pl = pl.rename(columns={lob_col: "LOB", "product_line": "Product line"})
            pl["Share (%)"] = (pl["Complaints"] / pl["Complaints"].sum() * 100).round(1)
            with st.expander("By product line (refinement within LOB)"):
                sort_options = ["Complaints", "Share (%)", "LOB", "Product line"]
                sort_by = st.selectbox(
                    "Sort product lines by",
                    sort_options,
                    index=0,
                    key="product_line_sort_by",
                )
                sort_desc = st.checkbox(
                    "Descending",
                    value=sort_by in {"Complaints", "Share (%)"},
                    key="product_line_sort_desc",
                )
                pl_view = pl.sort_values(by=sort_by, ascending=not sort_desc, kind="mergesort")
                st.dataframe(
                    pl_view,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Share (%)": st.column_config.NumberColumn("Share (%)", format="%.1f%%"),
                    },
                )
                st.markdown("<p class='caption-tufte'>E.g. Consumer Banking → Deposits vs Payments / money movement.</p>", unsafe_allow_html=True)
    else:
        st.info("No product data in selected range.")

    st.markdown("---")

    # ——— Issues ———
    st.markdown("## Issues")
    issue_col = "canonical_issue" if "canonical_issue" in df.columns else "issue"
    if issue_col in df.columns and not df.empty:
        issues = df[issue_col].value_counts().reset_index()
        issues.columns = ["Issue", "Complaints"]
        issues["Share (%)"] = (issues["Complaints"] / issues["Complaints"].sum() * 100).round(1)
        sort_options = ["Complaints", "Share (%)", "Issue"]
        issue_sort_by = st.selectbox(
            "Sort issues by",
            sort_options,
            index=0,
            key="issues_sort_by",
        )
        issue_sort_desc = st.checkbox(
            "Descending",
            value=issue_sort_by in {"Complaints", "Share (%)"},
            key="issues_sort_desc",
        )
        issues_view = issues.sort_values(by=issue_sort_by, ascending=not issue_sort_desc, kind="mergesort")
        st.dataframe(
            issues_view.head(20),
            use_container_width=True,
            hide_index=True,
            column_config={
                "Share (%)": st.column_config.NumberColumn("Share (%)", format="%.1f%%"),
            },
        )
        st.bar_chart(issues_view.head(20).set_index("Issue")["Complaints"])
        issue_pick = st.selectbox(
            "Download all complaints for issue",
            issues_view["Issue"].astype(str).tolist(),
            key="issues_download_pick",
        )
        issue_subset = df[df[issue_col].astype(str) == str(issue_pick)].copy()
        st.caption(f"{len(issue_subset)} complaints for selected issue.")
        issue_export = _prepare_complaint_export(issue_subset)
        safe_issue = "".join(ch if ch.isalnum() else "_" for ch in str(issue_pick)).strip("_")
        safe_issue = safe_issue[:80] if safe_issue else "issue"
        st.download_button(
            "Download complaints for selected issue (CSV)",
            issue_export.to_csv(index=False),
            file_name=f"issue_{safe_issue}_complaints.csv",
            mime="text/csv",
            key="issues_export",
        )
        st.markdown("<p class='caption-tufte'>Top 20 issues by volume (selected date range).</p>", unsafe_allow_html=True)
    else:
        st.info("No issue data in selected range.")

    st.markdown("---")

    # ——— Regions ———
    st.markdown("## Regions")
    if "state" in df.columns and not df.empty:
        states = df["state"].fillna("Unknown").value_counts().head(20).reset_index()
        states.columns = ["State", "Complaints"]
        states["% of total"] = (states["Complaints"] / states["Complaints"].sum() * 100).round(1).astype(str) + "%"
        st.dataframe(states, use_container_width=True, hide_index=True)
        st.bar_chart(states.set_index("State")["Complaints"])
        st.markdown("<p class='caption-tufte'>Complaints by state (top 20).</p>", unsafe_allow_html=True)
    else:
        st.info("No state/region data in selected range.")

    st.markdown("---")

    # ——— Hierarchy: LOB → Product line → Issue (precomputed windows) ———
    # Build hierarchy tied to selected window (current + prior) for consistency with filters
    hierarchy_respect_sidebar = st.checkbox(
        "Respect sidebar line-of-business filter in hierarchy chart",
        value=True,
        key="hierarchy_respect_sidebar",
    )
    df_for_hierarchy = df_full[
        (df_full["date_received"].dt.date >= prior_start.date())
        & (df_full["date_received"].dt.date <= end_d)
    ]
    if hierarchy_respect_sidebar and product_filter != "All lines of business":
        lob_col_full = "canonical_product" if "canonical_product" in df_for_hierarchy.columns else "product"
        df_for_hierarchy = df_for_hierarchy[df_for_hierarchy[lob_col_full].astype(str) == product_filter]
    hier_df = compute_hierarchy_summary(
        df_for_hierarchy,
        reference_date=pd.Timestamp(end_d),
        time_window_days=window_days,
        prior_window_days=window_days,
        seasonality_window_days=max(window_days * 3, 90),
    ) if not df_for_hierarchy.empty else pd.DataFrame()
    if not hier_df.empty:
        prod_col = "canonical_product" if "canonical_product" in hier_df.columns else "product"
        line_col = "product_line" if "product_line" in hier_df.columns else None
        issue_col = "canonical_issue" if "canonical_issue" in hier_df.columns else "issue"

        treemap_df = hier_df.copy()
        treemap_df["Product"] = treemap_df[prod_col].fillna("Unspecified").astype(str)
        treemap_df["Product line"] = treemap_df[line_col].fillna("Unspecified").astype(str) if line_col else "Unspecified"
        treemap_df["Issue"] = treemap_df[issue_col].fillna("Unspecified").astype(str)

        st.markdown("## Hierarchy (LOB → Product line → Issue)")
        scope_note = "Sidebar LOB filter applied." if hierarchy_respect_sidebar else "All lines of business shown."
        st.caption(f"Computed on selected window ({window_days}d) vs prior {window_days}d. {scope_note} {coverage_note}")

        fig = px.treemap(
            treemap_df,
            path=["Product", "Product line", "Issue"],
            values="current_volume",
            color="growth_ratio",
            color_continuous_scale="RdBu",
            color_continuous_midpoint=1.0,
            hover_data={"current_volume": True, "prior_volume": True, "growth_ratio": True},
        )
        st.plotly_chart(fig, use_container_width=True)

        # Nested expanders for drill-through
        for product in sorted(treemap_df["Product"].unique()):
            lob_rows = treemap_df[(treemap_df["Product"] == product) & (treemap_df["level"] == "lob")]
            lob_vol = int(lob_rows["current_volume"].iloc[0]) if not lob_rows.empty else 0
            with st.expander(f"{product} — {lob_vol} complaints (current window)", expanded=False):
                # Product lines
                if line_col:
                    lines = treemap_df[(treemap_df["Product"] == product) & (treemap_df["level"] == "product_line")]
                    if not lines.empty:
                        line_tbl = lines[[ "Product line", "current_volume", "prior_volume", "growth_ratio"]].rename(
                            columns={"current_volume": "Complaints", "prior_volume": "Prior", "growth_ratio": "Growth×"}
                        )
                        st.dataframe(line_tbl, use_container_width=True, hide_index=True)
                    # Issues within product
                    issues = treemap_df[(treemap_df["Product"] == product) & (treemap_df["level"] == "issue")]
                    if not issues.empty:
                        issue_tbl = issues[[ "Product line", "Issue", "current_volume", "prior_volume", "growth_ratio"]].rename(
                            columns={"current_volume": "Complaints", "prior_volume": "Prior", "growth_ratio": "Growth×"}
                        )
                        st.dataframe(issue_tbl, use_container_width=True, hide_index=True)

                # Drill to underlying complaints for this LOB (respect filters checkbox)
                respect_filters_lob = st.checkbox(f"Use sidebar filters for {product}", value=True, key=f"lob_filter_{product}")
                base_df = df if respect_filters_lob else df_for_hierarchy
                subset = base_df[(base_df.get("canonical_product", base_df.get("product")).astype(str) == product)]
                st.caption(f"Underlying complaints for {product} ({len(subset)} rows)")
                cols_show = ["date_received", "state", "canonical_issue", "issue", "product_line", "sub_product", "sub_issue", "company_response_to_consumer", "consumer_complaint_narrative"]
                cols_show = [c for c in cols_show if c in subset.columns]
                if subset.empty:
                    st.info("No complaints match the current selection.")
                else:
                    out = _prepare_complaint_export(subset)
                    out_display = out.copy()
                    out_display = out_display[[c for c in cols_show if c in out_display.columns]]
                    if "consumer_complaint_narrative" in out_display.columns:
                        out_display["consumer_complaint_narrative"] = out_display["consumer_complaint_narrative"].fillna("").astype(str).str.slice(0, 500)
                    st.dataframe(out_display, use_container_width=True, hide_index=True)
                    st.download_button(
                        f"Download complaints for {product}",
                        out.to_csv(index=False),
                        file_name=f"complaints_{product}.csv",
                        mime="text/csv",
                        key=f"dl_{product}",
                    )
    else:
        st.info("Run `scripts/emerging_topics.py` to build hierarchy (emerging_hierarchy.parquet).")

    st.markdown("---")

    # ——— Emerging topics (product-level; meaningful volume only) ———
    st.markdown("## Emerging topics (by product)")
    emergence_cfg = cfg.get("emergence", {})
    min_current_volume = int(emergence_cfg.get("min_current_volume", 10))
    min_prior_volume = int(emergence_cfg.get("min_prior_volume", 5))

    # Recompute emerging on the fly for the selected window
    df_for_emerging = df_full[(df_full["date_received"].dt.date >= prior_start.date()) & (df_full["date_received"].dt.date <= end_d)]
    emerging_records, emerging_summary = compute_emerging_products(
        df_for_emerging,
        time_window_days=window_days,
        prior_window_days=window_days,
        growth_threshold=float(emergence_cfg.get("growth_threshold", 1.5)),
        top_n_rank=int(emergence_cfg.get("top_n_rank", 5)),
        top_n_emerging=int(emergence_cfg.get("top_n_emerging", 10)),
        reference_date=pd.Timestamp(end_d),
        min_current_volume=min_current_volume,
        min_prior_volume=min_prior_volume,
        seasonality_window_days=max(window_days * 3, 90),
        persistence_min_runs=int(emergence_cfg.get("persistence_min_runs", 1)),
        hide_new_until_persistent=bool(emergence_cfg.get("hide_new_until_persistent", False)),
        previous_emerging=None,
    )
    if not emerging_records:
        st.info("No emerging products for the selected window; adjust dates or thresholds.")
    else:
        # Table: Rank | Product | Total complaints | States | Previous rank
        display = []
        for i, item in enumerate(emerging_records, 1):
            display.append({
                "Rank": item.get("rank_current"),
                "Product": item.get("product", ""),
                "Total complaints": item.get("current_volume"),
                "States": item.get("distinct_states", "—"),
                "Previous rank": item.get("rank_prior"),
                "_product": item.get("product", ""),
                "_takeaway": item.get("takeaway", ""),
                "_growth_pct": item.get("growth_pct"),
                "_prior_volume": item.get("prior_volume"),
                "_top_issues": item.get("top_issues", []),
            })
        tbl = pd.DataFrame([{k: v for k, v in d.items() if not k.startswith("_")} for d in display])
        st.dataframe(tbl, use_container_width=True, hide_index=True)
        st.markdown(f"<p class='caption-tufte'>Product-level emergence: selected {window_days} days vs prior {window_days} days. {coverage_note}</p>", unsafe_allow_html=True)
        for i, item in enumerate(display, 1):
            with st.expander(f"Details: {item['Product']} ({item['Total complaints']} complaints)", expanded=(i <= 2)):
                if item.get("_growth_pct") is not None and isinstance(item["_growth_pct"], (int, float)) and item["_growth_pct"] == item["_growth_pct"]:
                    st.caption(f"Up {item['_growth_pct']:.0f}% vs prior {window_days} days")
                elif item.get("_prior_volume") == 0:
                    st.caption("New spike this period")
                if item.get("_takeaway"):
                    st.write(item["_takeaway"])
                top_issues = item.get("_top_issues") or []
                if top_issues:
                    st.caption("**Top issues in this product (current window)**")
                    st.dataframe(pd.DataFrame(top_issues).rename(columns={"issue": "Issue", "current_volume": "Complaints"}), use_container_width=True, hide_index=True)
                topic_product = item.get("_product", "")
                prod_col = "canonical_product" if "canonical_product" in df.columns else "product"
                subset = df[df[prod_col].astype(str) == topic_product]
                if subset.empty:
                    st.caption("No complaints in the selected date range for this product.")
                else:
                    st.caption(f"**Underlying complaints ({len(subset)})** — in selected date range")
                    cols_show = ["date_received", "state", "issue", "sub_product", "sub_issue", "company_response_to_consumer"]
                    cols_show = [c for c in cols_show if c in subset.columns]
                    out = subset[cols_show].copy()
                    out["date_received"] = pd.to_datetime(out["date_received"]).dt.strftime("%Y-%m-%d")
                    st.dataframe(out, use_container_width=True, hide_index=True)
                    if "consumer_complaint_narrative" in subset.columns:
                        narratives = subset["consumer_complaint_narrative"].dropna().astype(str)
                        if not narratives.empty:
                            st.caption("**Sample narrative (first 500 chars)**")
                            st.text(narratives.iloc[0][:500] + ("…" if len(narratives.iloc[0]) > 500 else ""))
        # Flatten for CSV: top_issues as string
        def _row_for_csv(item: dict) -> dict:
            d = {k: v for k, v in item.items() if k != "top_issues" and not isinstance(v, list)}
            d["top_issues"] = "; ".join(f"{x.get('issue', '')} ({x.get('current_volume', 0)})" for x in (item.get("top_issues") or []))
            return d
        emerging_df = pd.DataFrame([_row_for_csv(x) for x in emerging_records])
        st.download_button("Download emerging topics (CSV)", emerging_df.to_csv(index=False), file_name="emerging_topics.csv", mime="text/csv")

    st.markdown("---")

    # ——— Emerging from narratives ———
    st.markdown("## Emerging from narratives")
    narrative_emerging = load_narrative_emerging()
    narrative_topics_tbl = load_narrative_topics_table()
    if not narrative_emerging:
        st.info("Run `scripts/narrative_topics.py` to compute narrative emerging topics.")
    else:
        dashboard_cfg = cfg.get("dashboard", {})
        auto_refresh_narratives = bool(dashboard_cfg.get("auto_refresh_narratives_on_stale", True))
        refresh_timeout_seconds = int(dashboard_cfg.get("narrative_refresh_timeout_seconds", 1800))

        complaints_max = _max_date(df_full["date_received"]) if "date_received" in df_full.columns else None
        # Compare narrative freshness to complaint rows that actually contain narratives.
        complaints_narr_max = None
        if {"date_received", "consumer_complaint_narrative"}.issubset(df_full.columns):
            narr_mask = (
                df_full["consumer_complaint_narrative"].notna()
                & (df_full["consumer_complaint_narrative"].astype(str).str.strip() != "")
            )
            if narr_mask.any():
                complaints_narr_max = _max_date(df_full.loc[narr_mask, "date_received"])
        freshness_target = complaints_narr_max or complaints_max
        narr_min = None
        narr_max = None
        narrative_respect_filters = True
        if not narrative_topics_tbl.empty and "date_received" in narrative_topics_tbl.columns:
            narrative_topics_tbl = narrative_topics_tbl.copy()
            narrative_topics_tbl["date_received"] = pd.to_datetime(
                narrative_topics_tbl["date_received"],
                errors="coerce",
            )
            narr_min = narrative_topics_tbl["date_received"].min()
            narr_max = narrative_topics_tbl["date_received"].max()

        stale_narratives = bool(
            freshness_target is not None
            and (narr_max is None or pd.isna(narr_max) or narr_max.date() < freshness_target.date())
        )
        refresh_target = freshness_target.strftime("%Y-%m-%d") if freshness_target is not None else "unknown"

        # Auto-refresh once per complaint max date to avoid repeated expensive runs on each rerun.
        if stale_narratives and auto_refresh_narratives:
            last_attempt = st.session_state.get("narrative_auto_refresh_attempt_for")
            if last_attempt != refresh_target:
                st.session_state["narrative_auto_refresh_attempt_for"] = refresh_target
                with st.spinner("Narrative outputs are stale; refreshing narrative pipeline..."):
                    ok, logs = _run_narrative_pipeline(timeout_seconds=refresh_timeout_seconds)
                if ok:
                    narrative_emerging = load_narrative_emerging()
                    narrative_topics_tbl = load_narrative_topics_table()
                    if not narrative_topics_tbl.empty and "date_received" in narrative_topics_tbl.columns:
                        narrative_topics_tbl = narrative_topics_tbl.copy()
                        narrative_topics_tbl["date_received"] = pd.to_datetime(
                            narrative_topics_tbl["date_received"],
                            errors="coerce",
                        )
                        narr_min = narrative_topics_tbl["date_received"].min()
                        narr_max = narrative_topics_tbl["date_received"].max()
                    stale_narratives = bool(
                        freshness_target is not None
                        and (narr_max is None or pd.isna(narr_max) or narr_max.date() < freshness_target.date())
                    )
                    st.success("Narrative outputs refreshed automatically.")
                else:
                    st.warning("Automatic narrative refresh failed. You can retry with the button below.")
                    if logs:
                        with st.expander("Refresh logs"):
                            st.code(logs)

        if pd.notna(narr_min) and pd.notna(narr_max):
            st.caption(
                "Narrative data coverage: "
                f"{narr_min.strftime('%Y-%m-%d')} to {narr_max.strftime('%Y-%m-%d')}"
            )
        if stale_narratives and freshness_target is not None:
            st.warning(
                "Narrative outputs are older than complaint data "
                f"(narratives through {narr_max.strftime('%Y-%m-%d') if pd.notna(narr_max) else '—'}, "
                f"narrative-capable complaints through {freshness_target.strftime('%Y-%m-%d')})."
            )
            if st.button("Refresh narrative outputs now", key="manual_refresh_narratives"):
                with st.spinner("Refreshing narrative pipeline..."):
                    ok, logs = _run_narrative_pipeline(timeout_seconds=refresh_timeout_seconds)
                if ok:
                    st.success("Narrative outputs refreshed.")
                    st.rerun()
                else:
                    st.error("Refresh failed.")
                    if logs:
                        with st.expander("Refresh logs"):
                            st.code(logs)

        narrative_respect_filters = st.checkbox(
            "Respect sidebar date and line-of-business filters for narrative drill-through",
            value=True,
            key="narrative_respect_filters",
        )
        narrative_topic_list_respect_sidebar = st.checkbox(
            "Apply sidebar date and line-of-business filters to narrative topic list",
            value=True,
            key="narrative_topic_list_respect_sidebar",
        )

        df_indexed = df_full.reset_index().rename(columns={"index": "complaint_idx"})
        filtered_narrative_emerging = narrative_emerging
        narrative_window_days_display = int(emergence_cfg.get("time_window_days", 30))
        if narrative_topic_list_respect_sidebar:
            topic_labels = _build_topic_label_map(narrative_emerging, narrative_topics_tbl)

            filtered_narrative_emerging = _recompute_narrative_emerging_for_sidebar(
                narrative_topics_tbl=narrative_topics_tbl,
                df_indexed=df_indexed,
                product_filter=product_filter,
                start_d=start_d,
                end_d=end_d,
                window_days=window_days,
                emergence_cfg=emergence_cfg,
                topic_labels=topic_labels,
            )
            narrative_window_days_display = window_days
            if product_filter == "All lines of business":
                st.caption(
                    f"{len(filtered_narrative_emerging)} narrative topics in selected window."
                )
            else:
                st.caption(
                    f"{len(filtered_narrative_emerging)} narrative topics match selected LOB: {product_filter}."
                )

        if not filtered_narrative_emerging:
            if narrative_topic_list_respect_sidebar:
                st.info("No narrative topics match the selected sidebar filters.")
            else:
                st.info("No narrative topics available.")

        for i, item in enumerate(filtered_narrative_emerging, 1):
            label = item.get("topic_label", f"Topic {item.get('topic_id', i)}")
            takeaway = item.get("takeaway", "")
            cur = item.get("current_volume")
            prior = item.get("prior_volume")
            with st.expander(label, expanded=(i <= 2)):
                if cur is not None and prior is not None:
                    if prior > 0:
                        pct = (cur / prior - 1) * 100
                        if pct == pct:
                            st.caption(f"Up {pct:.0f}% vs prior {narrative_window_days_display} days")
                        else:
                            st.caption("New or rising volume vs prior period")
                    else:
                        st.caption("New spike in narrative volume this period")
                st.write(takeaway)
                # Underlying complaints for this topic (respect current date/product filters)
                if not narrative_topics_tbl.empty:
                    topic_id = item.get("topic_id")
                    if topic_id is not None:
                        subset_topics = narrative_topics_tbl[narrative_topics_tbl["topic_id"] == topic_id]
                        if not subset_topics.empty:
                            # Align with filtered complaints
                            # Use full dataset index so complaint_idx matches pipeline output; keep only join keys to avoid column suffixes
                            subset_topics = subset_topics[["complaint_idx", "topic_id"]] if {"complaint_idx", "topic_id"}.issubset(subset_topics.columns) else subset_topics
                            merged_all = subset_topics.merge(df_indexed, on="complaint_idx", how="inner")
                            merged_all["date_received"] = pd.to_datetime(merged_all["date_received"])
                            merged = merged_all.copy()
                            if narrative_respect_filters:
                                merged = merged[
                                    (merged["date_received"].dt.date >= start_d)
                                    & (merged["date_received"].dt.date <= end_d)
                                ]
                                if product_filter != "All lines of business":
                                    lob_col = "canonical_product" if "canonical_product" in merged.columns else "product"
                                    merged = merged[merged[lob_col].astype(str) == product_filter]
                            if narrative_respect_filters:
                                st.caption(
                                    f"Underlying complaints ({len(merged)} shown of {len(merged_all)} for this topic)"
                                )
                            else:
                                st.caption(f"Underlying complaints ({len(merged)} for this topic)")
                            cols_show = [
                                "date_received",
                                "canonical_product",
                                "canonical_issue",
                                "product",
                                "issue",
                                "state",
                                "product_line",
                                "consumer_complaint_narrative",
                            ]
                            cols_show = [c for c in cols_show if c in merged.columns]
                            if merged.empty:
                                if narrative_respect_filters and not merged_all.empty:
                                    st.info(
                                        "No complaints match the current sidebar filters for this topic. "
                                        "Turn off the filter toggle above to view all topic complaints."
                                    )
                                else:
                                    st.info("No complaints found for this topic.")
                            # Build one comprehensive export (raw + enhanced + topic/issue context).
                            # If sidebar filters hide all rows, export full topic scope so users can still investigate.
                            export_source = merged if not merged.empty else merged_all
                            out_full = export_source.copy()
                            out_full["narrative_topic_id"] = topic_id
                            out_full["narrative_topic_label"] = label
                            out_full["narrative_current_volume"] = cur
                            out_full["narrative_prior_volume"] = prior
                            out_full["narrative_growth_ratio"] = (cur / prior) if (prior and prior > 0) else None

                            issue_group_col = None
                            if "canonical_issue" in out_full.columns:
                                issue_group_col = "canonical_issue"
                            elif "issue" in out_full.columns:
                                issue_group_col = "issue"
                            if issue_group_col is not None and not out_full.empty:
                                issue_counts = (
                                    out_full.groupby(issue_group_col, dropna=False)
                                    .size()
                                    .reset_index(name="topic_issue_volume")
                                )
                                out_full = out_full.merge(issue_counts, on=issue_group_col, how="left")

                            out_full = _prepare_complaint_export(
                                out_full,
                                leading_cols=[
                                    "narrative_topic_id",
                                    "narrative_topic_label",
                                    "narrative_current_volume",
                                    "narrative_prior_volume",
                                    "narrative_growth_ratio",
                                    "topic_issue_volume",
                                ],
                            )

                            if not merged.empty:
                                out_display = out_full[[c for c in cols_show if c in out_full.columns]].copy()
                                if "consumer_complaint_narrative" in out_display.columns:
                                    out_display["consumer_complaint_narrative"] = out_display["consumer_complaint_narrative"].fillna("").astype(str).str.slice(0, 500)
                                st.dataframe(out_display, use_container_width=True, hide_index=True)
                            elif narrative_respect_filters and not merged_all.empty:
                                st.caption("Download includes full topic scope because current sidebar filters return 0 rows.")

                            st.download_button(
                                "Download topic issues + complaint details (CSV)",
                                out_full.to_csv(index=False),
                                file_name=f"narrative_topic_{topic_id}_issues_and_details.csv",
                                mime="text/csv",
                                key=f"dl_topic_{topic_id}",
                            )
        narrative_df = pd.DataFrame(filtered_narrative_emerging)
        st.download_button("Download narrative emerging (CSV)", narrative_df.to_csv(index=False), file_name="narrative_emerging.csv", mime="text/csv", key="narrative_export")

    st.markdown("---")

    # ——— Hybrid search (keyword + semantic) ———
    st.markdown("## Hybrid search")
    st.markdown("<p class='caption-tufte'>Filter by LOB/issue (keyword), then optionally rank by semantic similarity to your query.</p>", unsafe_allow_html=True)
    import sys
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    paths_cfg = cfg.get("paths", {})
    embeddings_path = PROJECT_ROOT / paths_cfg.get("narrative_embeddings", "data/narrative_embeddings.parquet")
    try:
        from app.hybrid_search import (
            load_embeddings_table,
            hybrid_search,
            get_search_canonical_options,
        )
        from scripts.standardize_taxonomy import load_taxonomy
        tax = load_taxonomy(PROJECT_ROOT)
        canon_products, canon_issues = get_search_canonical_options(df_full, tax)
    except Exception as _e:
        canon_products, canon_issues = [], []
    emb_df = load_embeddings_table(embeddings_path) if embeddings_path.is_file() else pd.DataFrame()
    search_product = st.selectbox("Product (LOB)", ["Any"] + canon_products, key="hybrid_product")
    search_issue = st.selectbox("Issue", ["Any"] + canon_issues, key="hybrid_issue")
    query_text = st.text_input("Optional: free-text query (semantic similarity)", placeholder="e.g. wrong fee charged multiple times")
    run_search = st.button("Search complaints")
    res: pd.DataFrame | None = None
    semantic_used = False
    model_unavailable = False

    if run_search and not df_full.empty:
        prod = search_product if search_product != "Any" else None
        issue = search_issue if search_issue != "Any" else None
        model = None
        semantic_used = bool(query_text and query_text.strip())
        if semantic_used:
            try:
                from sentence_transformers import SentenceTransformer
                model = SentenceTransformer(cfg.get("embedding_model_id", "sentence-transformers/all-MiniLM-L6-v2"))
            except Exception:
                pass
        model_unavailable = semantic_used and model is None
        res = hybrid_search(
            df,  # use sidebar-filtered data (date/product) for coherence
            emb_df,
            query_text=query_text.strip() or None,
            canonical_product=prod,
            canonical_issue=issue,
            top_k=30,
            model=model,
            use_canonical=True,
        )
        st.session_state["hybrid_last_result"] = res.copy()
        st.session_state["hybrid_last_semantic_used"] = semantic_used
        st.session_state["hybrid_last_model_unavailable"] = model_unavailable
    elif run_search:
        st.info("Load complaint data first.")

    if res is None and isinstance(st.session_state.get("hybrid_last_result"), pd.DataFrame):
        res = st.session_state.get("hybrid_last_result")
        semantic_used = bool(st.session_state.get("hybrid_last_semantic_used", False))
        model_unavailable = bool(st.session_state.get("hybrid_last_model_unavailable", False))
        if not run_search:
            st.caption("Showing last search results.")

    if isinstance(res, pd.DataFrame):
        if res.empty:
            if semantic_used and model_unavailable:
                st.warning("Semantic model unavailable (torch/numpy issue). Showing no semantic results.")
            else:
                st.info("No complaints match the filters.")
        else:
            st.caption(f"**{len(res)}** complaints (keyword filter" + (" + semantic rank" if semantic_used else "") + ")")
            cols_show = [
                "date_received",
                "canonical_product" if "canonical_product" in res.columns else "product",
                "canonical_issue" if "canonical_issue" in res.columns else "issue",
                "state",
                "company",
                "sub_product",
                "sub_issue",
                "company_response_to_consumer",
                "submitted_via",
            ]
            if "similarity_score" in res.columns:
                cols_show = ["similarity_score"] + cols_show
            cols_show = [c for c in cols_show if c in res.columns]
            out = res[cols_show].head(50).copy()
            out["date_received"] = pd.to_datetime(out["date_received"]).dt.strftime("%Y-%m-%d")
            st.dataframe(out, use_container_width=True, hide_index=True)

            if "consumer_complaint_narrative" in res.columns:
                with st.expander("View full complaint narratives (top 10)", expanded=False):
                    narratives_df = res.head(10)[
                        [c for c in ["date_received", "canonical_product", "canonical_issue", "product", "issue", "state", "consumer_complaint_narrative"] if c in res.columns]
                    ].copy()
                    narratives_df["date_received"] = pd.to_datetime(narratives_df["date_received"]).dt.strftime("%Y-%m-%d")
                    st.dataframe(narratives_df, use_container_width=True, hide_index=True)
                    # Quick single selection to read one in full
                    opts = [
                        f"{i+1}. {narratives_df.iloc[i].get('canonical_product', narratives_df.iloc[i].get('product',''))} — {narratives_df.iloc[i].get('canonical_issue', narratives_df.iloc[i].get('issue',''))}"
                        for i in range(len(narratives_df))
                    ]
                    sel = st.selectbox("Read full narrative", ["Select..."] + opts, key="hybrid_narr_pick")
                    if sel != "Select...":
                        idx = opts.index(sel)
                        txt = str(narratives_df.iloc[idx]["consumer_complaint_narrative"])
                        st.text(txt)

            # Export full result rows (including narratives) for user verification
            search_export = _prepare_complaint_export(res.head(500))
            st.download_button(
                "Download search results (CSV)",
                search_export.to_csv(index=False),
                file_name="hybrid_search_results.csv",
                mime="text/csv",
                key="hybrid_export",
            )
    elif emb_df.empty or not canon_products:
        missing_steps: list[str] = []
        if not canon_products:
            missing_steps.append("<code>standardize_taxonomy.py</code>")
        if emb_df.empty:
            missing_steps.append("<code>narrative_topics.py</code>")
        hint = " and ".join(missing_steps) if missing_steps else "<code>standardize_taxonomy.py</code> and <code>narrative_topics.py</code>"
        st.markdown(
            f"<p class='caption-tufte'>Run {hint} for full hybrid search.</p>",
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # ——— Risk signals (systemic vs individual) ———
    st.markdown("## Risk signals: systemic vs individual")
    st.markdown("<p class='caption-tufte'>Product–issue buckets scored for systemic (process/operational) vs individual (e.g. pricing preference) risk. Systemic = higher regulatory/remediation priority.</p>", unsafe_allow_html=True)
    risk_path = PROJECT_ROOT / paths_cfg.get("risk_signals", "data/risk_signals.parquet")
    risk_table_respect_sidebar = st.checkbox(
        "Respect sidebar date and line-of-business filters in risk table",
        value=False,
        key="risk_table_respect_sidebar",
        help="If enabled, risk scores are recomputed from the currently filtered complaints.",
    )
    risk_df = pd.DataFrame()
    if risk_table_respect_sidebar:
        try:
            from scripts.standardize_taxonomy import load_taxonomy as _load_taxonomy
            from scripts.risk_signals import compute_product_issue_risk as _compute_risk
            _tax = _load_taxonomy(PROJECT_ROOT)
            risk_df = _compute_risk(df.copy(), _tax) if not df.empty else pd.DataFrame()
            st.caption("Risk table computed from sidebar-filtered complaints.")
        except Exception:
            st.warning("Could not recompute risk table from sidebar filters; falling back to precomputed file.")
            risk_df = pd.read_parquet(risk_path) if risk_path.is_file() else pd.DataFrame()
    elif risk_path.is_file():
        risk_df = pd.read_parquet(risk_path)

    if not risk_df.empty:
        risk_tier_filter = st.selectbox("Risk tier", ["All", "systemic", "elevated", "individual"], key="risk_tier")
        if risk_tier_filter != "All":
            risk_df = risk_df[risk_df["risk_tier"].astype(str) == risk_tier_filter]
        risk_display = risk_df[["canonical_product", "canonical_issue", "volume", "systemic_risk_score", "risk_tier", "risk_type"]].rename(
            columns={"canonical_product": "Product", "canonical_issue": "Issue", "volume": "Complaints", "systemic_risk_score": "Systemic score", "risk_tier": "Tier", "risk_type": "Risk type"}
        )
        st.dataframe(risk_display.head(50), use_container_width=True, hide_index=True)
        # Drill into underlying complaints for a selected product/issue
        if not risk_df.empty and {"canonical_product", "canonical_issue"}.issubset(risk_df.columns):
            options = (
                risk_df[["canonical_product", "canonical_issue", "risk_tier", "systemic_risk_score"]]
                .drop_duplicates()
                .assign(label=lambda d: d["canonical_product"].astype(str) + " — " + d["canonical_issue"].astype(str) + " (tier " + d["risk_tier"].astype(str) + ", score " + d["systemic_risk_score"].round(2).astype(str) + ")")
            )
            choice = st.selectbox("View underlying complaints", ["Select..."] + options["label"].tolist(), key="risk_drill_select")
            # Executives expect to see the selected time window by default
            respect_filters = st.checkbox("Respect sidebar date/product filters", value=True, help="If off, shows all complaints for this product/issue.")
            if choice != "Select...":
                sel = options.loc[options["label"] == choice].iloc[0]
                prod_sel, issue_sel = sel["canonical_product"], sel["canonical_issue"]
                base_df = df if respect_filters else df_full
                subset = base_df[
                    (base_df.get("canonical_product", base_df.get("product")).astype(str) == str(prod_sel))
                    & (base_df.get("canonical_issue", base_df.get("issue")).astype(str) == str(issue_sel))
                ]
                st.caption(f"Underlying complaints for {prod_sel} — {issue_sel} ({len(subset)} rows)")
                cols_show = [
                    "date_received",
                    "state",
                    "issue",
                    "canonical_product",
                    "canonical_issue",
                    "risk_type",
                    "sub_product",
                    "sub_issue",
                    "company_response_to_consumer",
                    "consumer_complaint_narrative",
                ]
                cols_show = [c for c in cols_show if c in subset.columns]
                if subset.empty:
                    st.info("No complaints match the current filters for this risk signal.")
                else:
                    out = _prepare_complaint_export(subset)
                    out_display = out.copy()
                    out_display = out_display[[c for c in cols_show if c in out_display.columns]]
                    if "consumer_complaint_narrative" in out_display.columns:
                        out_display["consumer_complaint_narrative"] = out_display["consumer_complaint_narrative"].fillna("").astype(str).str.slice(0, 500)
                    st.dataframe(out_display, use_container_width=True, hide_index=True)
                    st.download_button(
                        "Download underlying complaints (CSV)",
                        out.to_csv(index=False),
                        file_name=f"risk_underlying_{prod_sel}_{issue_sel}.csv",
                        mime="text/csv",
                    )
        st.download_button("Download risk signals (CSV)", risk_df.to_csv(index=False), file_name="risk_signals.csv", mime="text/csv", key="risk_export")
    else:
        st.info("No risk signals for the current mode/filters. Run `scripts/risk_signals.py` for precomputed risk signals.")

    st.markdown("---")

    # ——— Volume over time (Tufte: caption as explanation) ———
    st.markdown("## Complaints over time")
    if not df.empty and "date_received" in df.columns:
        df_chart = df.copy()
        df_chart["Week"] = df_chart["date_received"].dt.to_period("W").astype(str)
        weekly = df_chart.groupby("Week", dropna=False).size().reset_index(name="Complaints")
        st.bar_chart(weekly.set_index("Week")["Complaints"])
        st.markdown("<p class='caption-tufte'>Weekly complaint volume (selected date range and filters).</p>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
