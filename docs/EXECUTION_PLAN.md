# SmogAlert_PK → SmogNet Datathon: Execution Plan

**Status**: APPROVED — ready for execution  
**Date Created**: 2026-05-21  
**Purpose**: Upgrade SmogAlert_PK to meet all SmogNet Datathon (UET Mardan) requirements

---

## Context

The SmogNet Datathon requires a 3-stage end-to-end pollution intelligence pipeline built on a specific Kaggle dataset. The current project uses the wrong dataset (OpenAQ API, PM2.5-only, single unknown city) and is missing Stage 2 entirely (source classification). The architecture is sound and will be modified — not rebuilt from scratch.

**Datathon dataset**: `kaggle.com/datasets/hajramohsin/pakistan-air-quality-pollutant-concentrations`  
**Required pipeline**: Raw data → anomaly detection → source classification → alert generation → output

---

## Key Decisions (Pre-Approved)

1. **Source classifier**: Rule-based (Option A) — use chemical fingerprint thresholds, interpretable and fast
2. **Kaggle dataset**: Must be manually downloaded if not already present in `data/`
3. **Alert language**: Keep both English + Urdu
4. **Random Forest AQI classifier**: Keep as-is (doesn't conflict, adds value)

---

## Phase Overview

```
Phase 0: Dataset Acquisition & Audit
Phase 1: Data Layer Rebuild
Phase 2: Stage 1 — City+Season-Aware Anomaly Detection
Phase 3: Stage 2 — Source Classification (NEW MODULE)
Phase 4: Stage 3 — Enhanced Alert Generation
Phase 5: Dashboard Upgrade
Phase 6: End-to-End Integration & Output Validation
```

---

## Phase 0 — Dataset Acquisition & Audit

**Goal**: Get the correct Kaggle dataset loaded and fully understood before touching any code.

### Steps
1. Download Kaggle dataset: `hajramohsin/pakistan-air-quality-pollutant-concentrations`
2. Place CSV(s) in `data/` as `data/pakistan_aq_raw.csv`
3. Inspect dataset:
   - Exact column names for all 8 pollutants
   - Which cities are present, row count per city
   - Date range — confirm Jul–Dec 2024 test window exists
   - Missing value profile per pollutant per city
   - Time resolution (truly hourly?)
4. Confirm train/test split boundary: before 2024-07-01 = train; 2024-07-01 to 2024-12-31 = test

### Completion Checkpoint
Can answer: "What are all 8 column names? Which cities exist? Does Jul–Dec 2024 test window exist?"

---

## Phase 1 — Data Layer Rebuild

**Goal**: Replace wrong data source, expand preprocessing to handle 8 pollutants, multiple cities, train/test split.

### Files Changed
| File | Action |
|---|---|
| `download_data.py` | Rewrite — drop OpenAQ API, replace with Kaggle CSV loader |
| `src/preprocess.py` | Major rewrite — 8 pollutants, multi-city, seasonal features, split column |

### `download_data.py` Changes
- Remove all OpenAQ API and ZIP fallback logic
- New function: `load_kaggle_dataset()` — reads `data/pakistan_aq_raw.csv`, validates required columns, prints summary
- Saves copy as `data/raw_aqi_data.csv` to preserve downstream file references

### `src/preprocess.py` — New Output Schema
```
timestamp, city, pm25, pm10, no, no2, so2, nh3, co, o3,
hour, day_of_week, month, season, is_weekend,
pm25_24h_avg, pm10_24h_avg,
aqi_category, split
```

### New Preprocessing Steps
1. Parse and standardize timestamp
2. Handle missing values per-pollutant (ffill then bfill)
3. Filter invalid readings (negative values) per pollutant
4. Extract time features: `hour`, `day_of_week`, `month`, `is_weekend`
5. **NEW**: Add `season` column:
   - Winter: months 11, 12, 1, 2
   - Spring: months 3, 4, 5
   - Summer: months 6, 7, 8, 9
   - Autumn: month 10
6. **NEW**: Add `city_season` composite column (e.g., `Lahore_Winter`)
7. Compute 24h rolling averages for PM2.5 and PM10
8. Categorize AQI from PM2.5 (4-tier: Good/Moderate/Unhealthy/Hazardous)
9. **NEW**: Add `split` column — `'train'` if timestamp < 2024-07-01, else `'test'`
10. Save `data/cleaned_data.csv`

### Completion Checkpoint
`data/cleaned_data.csv` has multiple named cities, 8 pollutant columns, `season`, `city_season`, `split` columns, and Jul–Dec 2024 test rows are present.

---

## Phase 2 — Stage 1: City + Season Aware Anomaly Detection

**Goal**: Replace global single-city Isolation Forest with context-aware spike detector.

### Files Changed
| File | Action |
|---|---|
| `src/model.py` | Rewrite anomaly detection section (Functions 9–13 / Part 2) |
| `models/isolation_forest_*.pkl` | One model file per city_season group (replaces single file) |
| `outputs/anomalies.csv` | Regenerated with city, season, all pollutant columns |
| `outputs/anomaly_plot.png` | Regenerated — multi-city subplots |

### Design: Per-Group Isolation Forest
- Train one Isolation Forest **per `city_season` group** using **only training split** data
- Skip groups with fewer than 100 rows
- Save models as `models/isolation_forest_{city_season}.pkl`
- Apply each model to both train and test rows of that group
- Combine predictions into single DataFrame

### Features per group model
- `pm25`, `pm10`, `pm25_24h_avg`, `hour`

### Contamination
- Default `0.05` (5%) — applied locally per group, so contextually calibrated

### Output: `outputs/anomalies.csv`
Rows where `is_anomaly=1`, with all 8 pollutant columns preserved (required by Phase 3).

### Completion Checkpoint
`outputs/anomalies.csv` contains anomalous rows from multiple named cities, all 8 pollutant columns present, anomalies present from Jul–Dec 2024 test window.

---

## Phase 3 — Stage 2: Source Classification (NEW MODULE)

**Goal**: Classify each detected anomaly's probable emission source using chemical fingerprints.

### Files Changed
| File | Action |
|---|---|
| `src/source_classifier.py` | **NEW FILE** — entire Stage 2 |
| `outputs/anomalies_classified.csv` | **NEW OUTPUT** — anomalies + source label |

### Approach: Rule-Based Chemical Fingerprinting

Z-scores computed **per city-season group** using training data statistics (mean + std per pollutant).

| Source Label | Rule |
|---|---|
| `dust_storm` | PM10/PM2.5 ratio > 3.0 |
| `industrial` | SO₂ z-score > 1.5 |
| `crop_burning` | NH₃ z-score > 1.5 AND CO z-score > 1.5 |
| `vehicular` | NO z-score > 1.5 AND NO₂ z-score > 1.5 |
| `mixed` | Two or more of the above conditions are true |
| `unclassified` | None of the above thresholds met |

Rules applied in priority order: dust_storm → industrial → crop_burning → vehicular → mixed → unclassified.

### Output Columns Added
- `source_label`: one of the 6 labels above
- `source_confidence`: 0–1 score based on threshold exceedance magnitude
- `pollutant_signature`: human-readable string e.g. `"High NH₃ + CO"`

### Completion Checkpoint
`outputs/anomalies_classified.csv` has `source_label` column with at least 3 distinct source types present. All test-window anomalies have non-null source labels.

---

## Phase 4 — Stage 3: Enhanced Alert Generation

**Goal**: Generate structured 3–4 sentence public health alerts driven by source classification output.

### Files Changed
| File | Action |
|---|---|
| `src/alert_system.py` | Major upgrade — alerts now driven by source label + template library |
| `outputs/alerts_log.csv` | Regenerated with new full schema |

### New Alert Schema
```
timestamp, city, aqi_level, source_label, pollutant_signature,
affected_groups, protective_actions, alert_text_en, alert_text_ur, split
```

### Alert Template Library (one per source type)

**crop_burning**:
> "⚠ Air Quality Alert — [CITY]. Elevated levels of ammonia and carbon monoxide indicate active crop burning in or near the area, causing hazardous smog conditions. Children, elderly residents, and individuals with asthma or respiratory conditions are at highest risk. Avoid outdoor activity, keep windows closed, and use an N95 mask if you must go outside."

**vehicular**:
> "⚠ Air Quality Alert — [CITY]. High nitrogen oxide levels suggest heavy vehicular traffic is contributing to dangerous air pollution. Children, the elderly, and those with lung or heart conditions should limit outdoor exposure. Avoid high-traffic areas, use public transport if possible, and wear a mask outdoors."

**industrial**:
> "⚠ Air Quality Alert — [CITY]. Elevated sulfur dioxide readings point to industrial emissions as a probable cause of the current pollution spike. Respiratory patients, children, and elderly individuals are especially vulnerable. Stay indoors, close ventilation, and seek medical attention if experiencing breathing difficulty."

**dust_storm**:
> "⚠ Air Quality Alert — [CITY]. A high ratio of coarse to fine particulates suggests an active dust storm is reducing air quality. People with respiratory conditions, children, and the elderly should remain indoors with windows and doors sealed. Avoid all outdoor activity until conditions improve."

**mixed / unclassified**:
> "⚠ Air Quality Alert — [CITY]. Pollution levels have spiked above safe thresholds due to a combination of emission sources in the area. All sensitive groups — including children, elderly, and those with respiratory conditions — should minimize outdoor activity and wear protective masks."

### Alert Generation Criteria
- Only for anomalies in **test split** (Jul–Dec 2024)
- Only where AQI level is Unhealthy or Hazardous
- Target: 5–10 well-formed alerts per city

### Completion Checkpoint
`outputs/alerts_log.csv` has non-null `source_label`, `affected_groups`, `protective_actions`. At least 3 source types represented. Alerts are 3–4 sentences, non-technical, city-named.

---

## Phase 5 — Dashboard Upgrade

**Goal**: Update Streamlit dashboard to expose multi-city data, source classification results, enhanced alerts.

### Files Changed
| File | Action |
|---|---|
| `dashboard/app.py` | Significant additions — 2–3 new sections/tabs |

### Section Changes

**Map (existing)**: Use real city names. Color-code markers by current AQI. Click shows latest alert.

**Anomaly Timeline (upgrade)**: Add city filter dropdown. Add season shading bands.

**Source Classification (NEW TAB)**:
- Bar chart: source label distribution across all anomalies
- Table: per-city breakdown of source types (rows = cities, cols = source labels, values = count)
- "Top events" table: most severe test-window anomalies with source label, city, timestamp, PM2.5

**Alerts (upgrade)**:
- Source cause badge (🌾 Crop Burning, 🚗 Vehicular, 🏭 Industrial, 🌪 Dust Storm)
- Expandable rows showing affected groups + protective actions
- Filter by city, source type, date range

### Completion Checkpoint
Dashboard loads without errors. City dropdown shows real city names. Source classification tab displays distribution chart. Alert table shows cause + affected groups.

---

## Phase 6 — End-to-End Integration & Output Validation

**Goal**: Run full pipeline top-to-bottom, validate all datathon deliverables are satisfied.

### Pipeline Execution Order
```bash
python download_data.py
python src/preprocess.py
python src/model.py
python src/source_classifier.py
python src/alert_system.py
streamlit run dashboard/app.py
```

### Datathon Deliverable Checklist
| Deliverable | Source File | Verified? |
|---|---|---|
| Spike detection module | `src/model.py` | [ ] |
| Source classification module | `src/source_classifier.py` | [ ] |
| Alert generation module | `src/alert_system.py` | [ ] |
| Pollution trend visualizations | `outputs/*.png` + dashboard | [ ] |
| Detected anomaly timeline | `outputs/anomaly_plot.png` | [ ] |
| Classification summaries | Dashboard source tab | [ ] |
| 5–10 generated alerts | `outputs/alerts_log.csv` | [ ] |
| End-to-end pipeline (runs top to bottom) | All scripts | [ ] |

---

## File Change Summary

| File | Change Type | Phase |
|---|---|---|
| `download_data.py` | Rewrite | 1 |
| `src/preprocess.py` | Major rewrite | 1 |
| `src/model.py` | Significant modification | 2 |
| `src/source_classifier.py` | **New file** | 3 |
| `src/alert_system.py` | Major upgrade | 4 |
| `dashboard/app.py` | Significant additions | 5 |
| `data/pakistan_aq_raw.csv` | New data file | 0 |
| `data/cleaned_data.csv` | Regenerated | 1 |
| `outputs/anomalies.csv` | Regenerated | 2 |
| `outputs/anomalies_classified.csv` | New file | 3 |
| `outputs/alerts_log.csv` | Regenerated | 4 |
| `models/isolation_forest_*.pkl` | Multiple new model files | 2 |
