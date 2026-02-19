# CFPB Wells Fargo Emerging Topics POC

A **business-ready** proof-of-concept that downloads CFPB consumer complaint data, filters for Wells Fargo, detects **emerging topics** (Product–Issue combinations with rising volume or rank), and surfaces them in a Streamlit dashboard. Suitable for actual business use and buy-in for a larger implementation.

## Goal

Enable a bank executive to see **emerging topics** in consumer complaints (for Wells Fargo) in a **consumable, actionable** way—without drowning in raw data.

## Requirements

- **Python 3.10+**
- **Mac or Windows** (pathlib used throughout; no Unix-only or Windows-only logic)

## Install (Mac and Windows)

1. Clone or download this repo and `cd` into the project root.

2. Create a virtual environment and install dependencies:

   **Mac / Linux:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

   **Windows (PowerShell or Command Prompt):**
   ```powershell
   python -m venv .venv
   .venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **(Optional)** First run will download the embedding model from Hugging Face if you enable the narrative pipeline (phase 2). Ensure you have network access.

## Configure

Edit **`config.yaml`** in the project root. Key settings:

- **cfpb.source_url**: CFPB bulk download URL (default: complaints.csv.zip).
- **cfpb.company_filter_pattern**: Company name substring (default: `"Wells Fargo"`).
- **emergence**: `time_window_days`, `prior_window_days`, `growth_threshold`, `top_n_rank`, `top_n_emerging`.
- **paths**: Paths for data directory, Parquet files, manifest, pipeline status, and emerging-topics output.

Do **not** hardcode business logic in scripts; keep it in config.

## Run

### 1. Download CFPB data and filter for Wells Fargo

From the project root (with venv activated):

**Mac / Linux:**
```bash
python scripts/download_cfpb.py
```

**Windows:**
```powershell
python scripts\download_cfpb.py
```

- Downloads the CFPB complaints zip from the URL in config.
- Validates schema and row counts; filters by company pattern; writes **`data/wells_fargo_complaints.parquet`**.
- Writes **`data/manifest.json`** (source URL, last refresh timestamp, row counts) and **`data/pipeline_status.json`** (last run, status, duration, stage).
- Exit code **0** on success, **1** on failure (e.g. network error, missing columns). Check logs and pipeline_status if it fails.

### 2. Compute emerging topics

**Mac / Linux:**
```bash
python scripts/emerging_topics.py
```

**Windows:**
```powershell
python scripts\emerging_topics.py
```

- Reads config and **`data/wells_fargo_complaints.parquet`** (or **`data/wells_fargo_complaints_standardized.parquet`** if present).
- Aggregates by **canonical** product (LOB) and time windows; computes growth and rank; flags emerging topics.
- Writes **`data/emerging_topics.json`** and **`data/emerging_summary.parquet`**.

### 2.5. (Recommended) Taxonomy and risk signals

**Mac / Linux:**
```bash
python scripts/standardize_taxonomy.py
python scripts/risk_signals.py
```

- **standardize_taxonomy.py:** Applies **`data/taxonomy.yaml`** (redress LOB, product_line, issue, risk_type) to complaints; writes **`data/wells_fargo_complaints_standardized.parquet`**. LOBs reflect who owns the complaint (e.g. Consumer Banking includes deposits + payments/money movement; money transfer is not its own LOB). See **`docs/TAXONOMY_REDRESS.md`** for design and customization.
- **risk_signals.py:** Computes **systemic vs individual** risk scores per product–issue (narrative phrases + risk_type + volume/spread); writes **`data/risk_signals.parquet`**. Dashboard section **“Risk signals”** shows systemic/elevated/individual tiers.

### 3. (Optional) Narrative pipeline — emerging from consumer narratives

**Mac / Linux:**
```bash
python scripts/narrative_topics.py
```

**Windows:**
```powershell
python scripts\narrative_topics.py
```

- Requires **sentence-transformers** and **bertopic** (in `requirements.txt`). First run downloads the Hugging Face embedding model (default: `sentence-transformers/all-MiniLM-L6-v2`).
- Reads complaints Parquet; embeds the **full complaint corpus** in batch (config: `narrative_embed_batch_size`) using narrative text when present and structured fallback text otherwise; stores embeddings in **`data/narrative_embeddings.parquet`** (with **complaint_idx** for hybrid search).
- Runs BERTopic on the **narrative subset** (no vector DB); aggregates topic counts by current/prior time window; applies same emergence logic (growth, rank); writes **`data/narrative_emerging.json`**.
- Dashboard sections **“Emerging from narratives”** and **“Hybrid search”** (keyword + semantic) use these embeddings when the file exists.

### 4. Launch the dashboard

**Mac / Linux:**
```bash
streamlit run app/dashboard.py
```

**Windows:**
```powershell
streamlit run app\dashboard.py
```

- Opens the Streamlit app in your browser.
- Shows **Data as of** / **Last refreshed**, row count, KPIs, **Emerging topics** (by canonical product), **Emerging from narratives** (when `narrative_topics.py` has been run), **Hybrid search** (keyword LOB/issue + optional semantic query), **Risk signals** (systemic vs individual), and charts (volume over time, top issues).
- If narrative outputs are stale vs complaint data, dashboard auto-refreshes the narrative pipeline once per data refresh (config: `dashboard.auto_refresh_narratives_on_stale`).
- One-click **export** of emerging topics, narrative emerging, and risk signals as CSV. Methodology is linked in the app.

## Refresh cadence

- CFPB updates the complaint database **daily**. Run **`download_cfpb.py`** at least daily (e.g. via cron or Task Scheduler), then **`emerging_topics.py`**, so the dashboard reflects the latest data.
- Re-run both scripts before a steering meeting to ensure “Data as of” is current.

## Runbook (what to do when things go wrong)

| Situation | Action |
|-----------|--------|
| Dashboard shows “No complaint data” | Run `download_cfpb.py` first; ensure `data/wells_fargo_complaints.parquet` exists. |
| Dashboard shows “No emerging topics” | Run `emerging_topics.py` after download; check `data/emerging_topics.json`. |
| Download fails (network / timeout) | Check CFPB URL in config; retry; check `data/pipeline_status.json` and logs for error. |
| CFPB URL or schema changed | Update **cfpb.source_url** in config; if columns changed, update **scripts/download_cfpb.py** (required columns: date_received, company, product, issue) and re-run. |
| Empty Wells Fargo filter | Confirm **company_filter_pattern** in config (e.g. `"Wells Fargo"`); CFPB company names may vary slightly. |
| No narrative emerging | Dashboard auto-refreshes narratives when stale (default). If still empty, run `narrative_topics.py` and ensure deps are installed (`sentence-transformers`, `bertopic`). Many complaints lack narratives; section may be empty. |
| Hybrid search empty or no similarity | Run `standardize_taxonomy.py` and `narrative_topics.py` so embeddings parquet has `complaint_idx` and canonical product/issue. |
| No risk signals | Run `standardize_taxonomy.py` then `risk_signals.py`; check `data/risk_signals.parquet`. |

## Vector and embedding strategy

- **Model source:** Hugging Face only; pre-trained sentence-transformers (no custom training). Default: **`sentence-transformers/all-MiniLM-L6-v2`** (384d); configurable via **`embedding_model_id`** in `config.yaml`.
- **When we embed:** During the narrative pipeline step (`narrative_topics.py`), after filtering to Wells Fargo. Every complaint gets an embedding (narrative text if present; structured fallback if missing narrative); batch size in config to avoid OOM.
- **Where we store:** Parquet (**`data/narrative_embeddings.parquet`**) with columns `complaint_idx`, `date_received`, `has_narrative`, and `embedding` (list of floats). No vector database for POC.
- **How we use:** Hybrid search ranks complaint rows by cosine similarity to query embedding. BERTopic uses the narrative subset for topic/cluster emergence over time.

## Optional upgrades (domain / Tier 2)

- **FinTextSim** (or **FinLang/finance-embeddings-investopedia**): Domain-specific financial embeddings can improve topic coherence on banking language. To use, set **`embedding_model_id`** in `config.yaml` to the Hugging Face model id (e.g. after validating the POC with the default model). Not used for one-shot.
- **FASTopic** (NeurIPS 2024): Alternative topic model; replace BERTopic in `narrative_topics.py` with FASTopic if you adopt it (integration effort higher). **BERTrend** (ACL 2024) and time-aware embeddings are other Tier 2 options for stronger emergence detection; document in config when you add them.

## Tests

From the project root:

**Mac / Linux:**
```bash
pytest tests/ -v
```

**Windows:**
```powershell
pytest tests\ -v
```

- **Emerging-topics logic**: fixture with known counts → assert correct items flagged as emerging and thresholds applied.
- **Download + filter**: mocked/small CSV zip → assert Wells Fargo filter and column presence.

## Methodology

See **[docs/METHODOLOGY.md](docs/METHODOLOGY.md)** for a one-page definition of “emerging,” time windows, thresholds, and data source. Use it for executives and compliance.

## Path to full implementation

The POC is step 1 of a larger program. For production scale you would typically:

- **Config**: Move to a parameter store or CI-managed config; keep the same structure.
- **Data**: Ingest into a data warehouse (e.g. Snowflake, BigQuery) with incremental loads; keep lineage (manifest) in a catalog.
- **Orchestration**: Run download and analysis via a scheduler (e.g. Airflow, Prefect) with retries and alerting on failure.
- **Dashboard**: Embed in enterprise BI (Tableau, Power BI) or keep Streamlit behind auth and scaling.
- **Testing**: Add integration tests (e.g. run pipeline on last 7 days in CI) and monitoring on freshness and row counts.

Logic and methodology (definition of “emerging,” time windows, thresholds) carry over; infrastructure is upgraded without redefining the business rules.

## Project layout

| Path | Purpose |
|------|--------|
| `config.yaml` | Source URL, company filter, time windows, emergence thresholds, paths |
| `requirements.txt` | Pinned or min versions for reproducibility |
| `scripts/download_cfpb.py` | Download CFPB zip, validate, filter, save Parquet, write manifest and pipeline_status |
| `scripts/emerging_topics.py` | Compute emerging topics; write JSON and optional summary Parquet |
| `app/dashboard.py` | Streamlit dashboard: lineage, KPIs, emerging topics, charts, export |
| `data/` | Parquet, manifest, pipeline_status, emerging_topics (gitignored or sample only) |
| `tests/` | Unit tests for emerging-topics and download+filter |
| `docs/METHODOLOGY.md` | One-page methodology for executives and compliance |

## License and data

- CFPB Consumer Complaint Database is public. See [consumerfinance.gov](https://www.consumerfinance.gov/data-research/consumer-complaints/) for data use and attribution.
- This POC code is provided as-is for internal use and buy-in; adjust license as needed for your organization.
