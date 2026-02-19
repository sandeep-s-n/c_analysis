# Methodology: Emerging Topics

This document defines how **emerging topics** are computed for the CFPB Wells Fargo Emerging Topics POC. It is intended for executives and compliance so the logic is transparent and auditable.

## Data source

- **CFPB Consumer Complaint Database**  
  Complaints are downloaded from the CFPB bulk file:  
  `https://files.consumerfinance.gov/ccdb/complaints.csv.zip`  
  The database is updated daily. Complaints are published after the company responds or after 15 days, whichever comes first.

- **Scope**  
  Only complaints where the **company** field contains **"Wells Fargo"** (case-insensitive) are included. This covers Wells Fargo Bank, WELLS FARGO COMPANY, and related subsidiaries as named in the CFPB data.

## Definition of “emerging”

Emergence is computed at **product level** (not product×issue) so volumes are meaningful (e.g. hundreds of complaints per product), avoiding noise from 2–3 complaint cells.

An **emerging product** is a **Product** (line of business) that meets at least one of the following **and** meets minimum volume bars:

1. **Volume growth**  
   Complaints in the **current time window** (e.g. last 30 days) are at least **X times** the volume in the **prior time window** (e.g. previous 30 days), **and** prior-period volume ≥ `min_prior_volume`.  
   Example: if the threshold is 1.5 and min_prior_volume is 5, then “Mortgage” is emerging when current-period volume ≥ 1.5 × prior-period volume and prior volume ≥ 5.

2. **Rising rank**  
   The product moved **up in rank** (by volume) compared to the prior period and is now in the **top N** (e.g. top 5) by volume in the current period.

**Minimum volume (noise filter):**  
Only products with **current-period volume ≥ `min_current_volume`** (e.g. 10) are surfaced. This avoids surfacing “emerging” topics with 2–3 complaints, which are not actionable.

Time windows and thresholds are set in **config** (e.g. `config.yaml`):  
`time_window_days`, `prior_window_days`, `growth_threshold`, `top_n_rank`, `top_n_emerging`, `min_current_volume`, `min_prior_volume`.

**Drill-down:** For each emerging product, the pipeline attaches **top issues** (issue-level counts in the current window) so users can see which issues drive the product-level trend.

## Time windows

- **Current window**: last `time_window_days` days (e.g. 30) before “now” (run time or reference date).  
- **Prior window**: the `prior_window_days` days (e.g. 30) immediately before the current window.

Complaints are bucketed by `date_received` (CFPB “Date received”). Only complaints in these two windows are used to compute current and prior volumes **per product**. The reference for “current” and “prior” is the **end of the available data** (max `date_received`), not run time, so results are stable for a given dataset.

## Output

- **Emerging list**: up to `top_n_emerging` **products**, sorted by growth ratio and volume, each with `current_volume`, `prior_volume`, `growth_pct`, rank change, **top_issues** (issue breakdown), and a short **takeaway**.
- **Summary table**: all products in the current and prior windows with volumes, growth ratio, ranks, and an “emerging” flag (optional Parquet/JSON for audit).

## Reproducibility

Given the same **config** and the same **CFPB data** (same download), the pipeline produces the same emerging-topics list. Python and key dependency versions are documented in `requirements.txt` and README.

## Taxonomy (canonical LOB and issues)

- **Source:** `data/taxonomy.yaml` maps raw CFPB **product** and **issue** values to **canonical_product** (LOB) and **canonical_issue** (display label), and assigns **risk_type** per issue: `operational`, `pricing_dispute`, `service`, `fraud`, `other`.
- **Standardization:** `scripts/standardize_taxonomy.py` applies the taxonomy to complaints and writes **`data/wells_fargo_complaints_standardized.parquet`** (same schema plus `canonical_product`, `canonical_issue`, `risk_type`). Emerging-topics and dashboard use standardized data when present so LOB/issue filters and emergence are on canonical values.

## Hybrid search (keyword + semantic)

- **Purpose:** Combine **keyword** filters (product/issue) with **semantic similarity** over full-corpus complaint embeddings so drill-down can mix structured filters and free-text query.
- **Flow:** Filter complaints by canonical product and/or issue; join to **narrative_embeddings.parquet** (by `complaint_idx`); if the user provides a free-text query, embed it with the same sentence-transformers model and rank filtered rows by cosine similarity to the query; return top-k complaints with optional `similarity_score`.
- **Data:** Requires `scripts/narrative_topics.py` to have been run (embeddings parquet with `complaint_idx`). Dashboard section “Hybrid search” exposes product/issue dropdowns and optional query box.

## Risk signals: systemic vs individual

- **Purpose (risk mitigation):** Distinguish **systemic** issues (process/systems failure, many affected, regulatory/remediation priority) from **individual** issues (e.g. “interest too high” — customer preference, lower systemic risk).
- **Signals:**  
  1. **Taxonomy risk_type:** `operational` issues (e.g. incorrect report, investigation problems) get higher systemic weight; `pricing_dispute` (e.g. fees/interest) gets lower (often individual).  
  2. **Narrative phrases:** Systemic phrases (e.g. “multiple times”, “same problem”, “their system”, “bank error”, “not just me”) vs individual (“interest too high”, “I think”, “my rate”). Each narrative is scored; product–issue buckets get mean narrative systemic score.  
  3. **Volume and spread:** Higher volume and more distinct states for a product–issue increase systemic score.
- **Composite:** `systemic_risk_score` (0–1) = 0.35 × mean narrative systemic + 0.35 × risk_type weight + 0.2 × geographic spread + 0.1 × volume (log norm). **Risk tier:** `systemic` (≥0.5), `elevated` (≥0.35), `individual` (<0.35).
- **Output:** `scripts/risk_signals.py` writes **`data/risk_signals.parquet`** (canonical_product, canonical_issue, volume, systemic_risk_score, risk_tier, risk_type). Dashboard section “Risk signals” shows the table and allows filter by risk tier.

## Narrative-based emerging (phase 2)

When the narrative pipeline is enabled, “emerging from narratives” uses the same **time-window and growth/rank logic** applied to **topic or cluster** counts derived from `consumer_complaint_narrative` (e.g. BERTopic or sentence-transformers + clustering). Only narrative topics with **current_volume ≥ min_current_volume** are surfaced, so low-volume (2–3 complaint) clusters are excluded.
