"""
Source Classification Module — SmogAlert PK (SmogNet Datathon)
==============================================================
Stage 2 of the 3-stage pipeline: classify each detected anomaly's
probable emission source using chemical fingerprinting.

How it works:
  1. Load training data → compute per-city-season mean and std for each pollutant
  2. Load anomalies detected in Stage 1 (outputs/anomalies.csv)
  3. For every anomaly row, compute z-scores (how many std-devs above normal?)
  4. Apply rule-based fingerprint rules to assign a source label
  5. Add source_label, source_confidence, pollutant_signature columns
  6. Save outputs/anomalies_classified.csv

Source Labels
-------------
  dust_storm   — high PM10 relative to PM2.5 (coarse dust dominates)
  industrial   — elevated SO2 (sulfur from fuel combustion / factories)
  crop_burning — elevated NH3 + CO (nitrogen from fertilized crop residue)
  vehicular    — elevated NO + NO2 (nitrogen oxides from engines)
  mixed        — two or more of the above conditions triggered
  unclassified — no threshold exceeded

Why rule-based?
  Interpretable and fast. Each rule maps to a known chemical process,
  making results easy to explain to non-technical judges and the public.
"""

import os
import pandas as pd
import numpy as np

# ============================================================================
# CONFIGURATION
# ============================================================================

DATA_DIR    = "data"
OUTPUTS_DIR = "outputs"

CLEANED_DATA_PATH = os.path.join(DATA_DIR,    "cleaned_data.csv")
ANOMALIES_PATH    = os.path.join(OUTPUTS_DIR, "anomalies.csv")
OUTPUT_PATH       = os.path.join(OUTPUTS_DIR, "anomalies_classified.csv")

# Pollutants whose z-scores are used in fingerprint rules
Z_SCORE_POLLUTANTS = ['so2', 'nh3', 'co', 'no', 'no2']

# Rule thresholds — how many std-devs above the group mean counts as "elevated"
Z_THRESHOLD = 1.5

# Dust rule threshold — PM10/PM2.5 ratio above this signals coarse dust
DUST_RATIO_THRESHOLD = 3.0

# Confidence score normalisation denominator:
#   z_score = 1.5  → raw excess = 0.0 → confidence ≈ 0.0
#   z_score = 4.5  → raw excess = 3.0 → confidence = 1.0
CONFIDENCE_SCALE = 3.0


# ============================================================================
# FUNCTION 1: Compute Per-Group Training Statistics
# ============================================================================

def compute_training_stats(cleaned_df):
    """
    Compute mean and standard deviation for each pollutant within every
    city-season group, using ONLY the training split.

    Why training split only?
      The z-score baseline must reflect 'normal' conditions that the model
      learned from. Using test data would leak future information into the
      classification step.

    Parameters:
        cleaned_df (pd.DataFrame): Full cleaned dataset (train + test rows).
                                   Must have columns: city_season, split,
                                   and all pollutants in Z_SCORE_POLLUTANTS.

    Returns:
        dict: Nested dict — stats[city_season][pollutant] = {'mean': ..., 'std': ...}
              Returns a dict for every city_season group that has training data.
    """
    # Keep only training rows for baseline statistics
    train_df = cleaned_df[cleaned_df['split'] == 'train'].copy()

    print(f"  Using {len(train_df):,} training rows to compute group baselines")

    stats = {}

    for group_name, group_df in train_df.groupby('city_season'):
        stats[group_name] = {}

        for pollutant in Z_SCORE_POLLUTANTS:
            if pollutant not in group_df.columns:
                continue

            group_values = group_df[pollutant].dropna()

            if len(group_values) < 10:
                # Too few values to compute a reliable baseline — skip
                continue

            group_mean = group_values.mean()
            group_std  = group_values.std()

            # Protect against zero std (constant column) — would cause division by zero
            # Use a small floor value to keep z-scores finite
            if group_std < 1e-6:
                group_std = 1e-6

            stats[group_name][pollutant] = {
                'mean': group_mean,
                'std':  group_std
            }

    print(f"  Computed baselines for {len(stats)} city-season groups")
    return stats


# ============================================================================
# FUNCTION 2: Compute Z-Scores for One Row
# ============================================================================

def compute_z_scores(row, stats):
    """
    For a single anomaly row, look up its city-season group baseline and
    compute how many std-devs above normal each pollutant reading is.

    Parameters:
        row (pd.Series): One row from the anomalies DataFrame.
                         Must have 'city_season' and pollutant columns.
        stats (dict):    Training stats from compute_training_stats().

    Returns:
        dict: {pollutant: z_score} for every pollutant in Z_SCORE_POLLUTANTS.
              Missing or uncomputable z-scores are stored as 0.0.
    """
    group_name = row.get('city_season', '')
    group_stats = stats.get(group_name, {})

    z_scores = {}

    for pollutant in Z_SCORE_POLLUTANTS:
        if pollutant not in group_stats:
            # No baseline for this group/pollutant — treat as normal (z=0)
            z_scores[pollutant] = 0.0
            continue

        value = row.get(pollutant, np.nan)

        if pd.isna(value):
            z_scores[pollutant] = 0.0
            continue

        mean = group_stats[pollutant]['mean']
        std  = group_stats[pollutant]['std']

        z_scores[pollutant] = (value - mean) / std

    return z_scores


# ============================================================================
# FUNCTION 3: Evaluate Individual Fingerprint Conditions
# ============================================================================

def evaluate_conditions(row, z_scores):
    """
    Check whether each chemical fingerprint condition is met for one row.

    Each condition corresponds to a known emission process:
      - Dust storm: coarse-to-fine ratio in particulates
      - Industrial: sulfur burning (power plants, brick kilns, refineries)
      - Crop burning: ammonia + carbon monoxide from burning fields
      - Vehicular: nitrogen oxides from internal combustion engines

    Parameters:
        row      (pd.Series): Anomaly row (needs 'pm25', 'pm10').
        z_scores (dict):      Pre-computed z-scores from compute_z_scores().

    Returns:
        dict: {condition_name: bool} — True if the condition's threshold is met.
    """
    # --- Dust Storm ---
    # PM10 (coarse particles) disproportionately high vs PM2.5 (fine particles).
    # Ratio > 3 means the air is full of large dust grains, not combustion smoke.
    pm25_val = row.get('pm25', np.nan)
    pm10_val = row.get('pm10', np.nan)

    if pd.notna(pm25_val) and pd.notna(pm10_val) and pm25_val > 0:
        dust_ratio = pm10_val / pm25_val
        cond_dust = dust_ratio > DUST_RATIO_THRESHOLD
    else:
        dust_ratio = 0.0
        cond_dust  = False

    # --- Industrial ---
    # SO2 z-score elevated → sulfur-rich fuel being burned (coal, diesel, heavy oil).
    cond_industrial = z_scores.get('so2', 0.0) > Z_THRESHOLD

    # --- Crop Burning ---
    # BOTH NH3 and CO elevated together → fertilizer-laden crop residue being burned.
    # NH3 alone can also come from livestock; requiring CO as co-indicator targets burning.
    cond_crop = (z_scores.get('nh3', 0.0) > Z_THRESHOLD and
                 z_scores.get('co',  0.0) > Z_THRESHOLD)

    # --- Vehicular ---
    # BOTH NO and NO2 elevated → combustion exhaust from petrol/diesel engines.
    cond_vehicular = (z_scores.get('no',  0.0) > Z_THRESHOLD and
                      z_scores.get('no2', 0.0) > Z_THRESHOLD)

    return {
        'dust_storm':   cond_dust,
        'industrial':   cond_industrial,
        'crop_burning': cond_crop,
        'vehicular':    cond_vehicular,
        '_dust_ratio':  dust_ratio      # stored for confidence calculation
    }


# ============================================================================
# FUNCTION 4: Assign Source Label
# ============================================================================

def assign_source_label(conditions):
    """
    Assign a single source label based on which conditions are triggered.

    Decision logic:
      - If 2+ individual conditions are true → 'mixed' (multiple sources)
      - If exactly 1 is true → use that source's name
      - If none → 'unclassified'

    The individual label priority (dust > industrial > crop > vehicular) only
    matters when exactly one condition is true; there can be no tie in that case.

    Parameters:
        conditions (dict): Output of evaluate_conditions() — bools per source.

    Returns:
        str: One of: dust_storm, industrial, crop_burning, vehicular,
                     mixed, unclassified
    """
    # Count how many of the four source conditions are true
    individual_flags = [
        conditions['dust_storm'],
        conditions['industrial'],
        conditions['crop_burning'],
        conditions['vehicular'],
    ]
    n_triggered = sum(individual_flags)

    if n_triggered >= 2:
        return 'mixed'
    elif conditions['dust_storm']:
        return 'dust_storm'
    elif conditions['industrial']:
        return 'industrial'
    elif conditions['crop_burning']:
        return 'crop_burning'
    elif conditions['vehicular']:
        return 'vehicular'
    else:
        return 'unclassified'


# ============================================================================
# FUNCTION 5: Compute Source Confidence Score
# ============================================================================

def compute_confidence(source_label, conditions, z_scores):
    """
    Produce a 0–1 confidence score reflecting how strongly the
    chemical fingerprint evidence supports the assigned label.

    Method: for each triggered condition, compute how far above the
    threshold the evidence is, then normalise to [0, 1].

    Parameters:
        source_label (str):   Label assigned by assign_source_label().
        conditions   (dict):  Condition booleans + _dust_ratio.
        z_scores     (dict):  Pollutant z-scores.

    Returns:
        float: Confidence in [0.0, 1.0]. Unclassified always returns 0.0.
    """
    if source_label == 'unclassified':
        return 0.0

    def _dust_confidence():
        ratio = conditions.get('_dust_ratio', 0.0)
        # Linear scale: threshold=3.0 → 0.0, threshold+SCALE=6.0 → 1.0
        excess = max(0.0, ratio - DUST_RATIO_THRESHOLD)
        return min(1.0, excess / CONFIDENCE_SCALE)

    def _z_confidence(pollutant):
        z = z_scores.get(pollutant, 0.0)
        excess = max(0.0, z - Z_THRESHOLD)
        return min(1.0, excess / CONFIDENCE_SCALE)

    def _pair_confidence(p1, p2):
        # For two-pollutant rules, use the weaker of the two (conservative)
        return min(_z_confidence(p1), _z_confidence(p2))

    label_confidence_map = {
        'dust_storm':   _dust_confidence(),
        'industrial':   _z_confidence('so2'),
        'crop_burning': _pair_confidence('nh3', 'co'),
        'vehicular':    _pair_confidence('no', 'no2'),
    }

    if source_label == 'mixed':
        # Average confidence across all triggered individual conditions
        triggered_scores = [
            score for lbl, score in label_confidence_map.items()
            if conditions.get(lbl, False)
        ]
        return float(np.mean(triggered_scores)) if triggered_scores else 0.0

    return label_confidence_map.get(source_label, 0.0)


# ============================================================================
# FUNCTION 6: Build Pollutant Signature String
# ============================================================================

def build_signature(source_label, conditions, z_scores, row):
    """
    Build a human-readable string summarising the chemical evidence that
    led to the source label — e.g. 'High NH₃ + CO', 'PM10/PM2.5 ratio: 4.2'.

    Used for display in the dashboard and alert text.

    Parameters:
        source_label (str):   Assigned source label.
        conditions   (dict):  Condition booleans.
        z_scores     (dict):  Pollutant z-scores.
        row          (pd.Series): Original anomaly row (needs pm10, pm25).

    Returns:
        str: One-line description of key chemical indicators.
    """
    if source_label == 'unclassified':
        # Report the highest z-score pollutant even if below threshold
        best_pollutant = max(z_scores, key=lambda p: z_scores.get(p, 0.0))
        best_z = z_scores.get(best_pollutant, 0.0)
        return f"Highest z-score: {best_pollutant.upper()} ({best_z:.1f}σ)"

    parts = []

    if conditions['dust_storm']:
        ratio = conditions.get('_dust_ratio', 0.0)
        parts.append(f"PM10/PM2.5 ratio: {ratio:.1f}")

    if conditions['industrial']:
        z = z_scores.get('so2', 0.0)
        parts.append(f"High SO₂ ({z:.1f}σ)")

    if conditions['crop_burning']:
        z_nh3 = z_scores.get('nh3', 0.0)
        z_co  = z_scores.get('co',  0.0)
        parts.append(f"High NH₃ ({z_nh3:.1f}σ) + CO ({z_co:.1f}σ)")

    if conditions['vehicular']:
        z_no  = z_scores.get('no',  0.0)
        z_no2 = z_scores.get('no2', 0.0)
        parts.append(f"High NO ({z_no:.1f}σ) + NO₂ ({z_no2:.1f}σ)")

    return "; ".join(parts) if parts else "—"


# ============================================================================
# FUNCTION 7: Classify a Single Anomaly Row (combines 3–6)
# ============================================================================

def classify_row(row, stats):
    """
    End-to-end classification for one anomaly row:
      1. Compute z-scores against training baseline
      2. Evaluate fingerprint conditions
      3. Assign source label
      4. Compute confidence
      5. Build signature string

    Parameters:
        row   (pd.Series): One row from anomalies DataFrame.
        stats (dict):      Training stats from compute_training_stats().

    Returns:
        pd.Series: Three new values — source_label, source_confidence,
                   pollutant_signature.
    """
    z_scores   = compute_z_scores(row, stats)
    conditions = evaluate_conditions(row, z_scores)
    label      = assign_source_label(conditions)
    confidence = compute_confidence(label, conditions, z_scores)
    signature  = build_signature(label, conditions, z_scores, row)

    return pd.Series({
        'source_label':        label,
        'source_confidence':   round(confidence, 3),
        'pollutant_signature': signature
    })


# ============================================================================
# FUNCTION 8: Print Classification Summary
# ============================================================================

def print_classification_summary(classified_df):
    """
    Print a concise summary of classification results split by
    train/test window and source label.

    Parameters:
        classified_df (pd.DataFrame): Output DataFrame with source_label column.
    """
    print("\n" + "=" * 70)
    print("CLASSIFICATION SUMMARY")
    print("=" * 70)

    total = len(classified_df)
    print(f"\nTotal anomalies classified: {total:,}")

    # Overall label distribution
    print("\nSource label distribution (all anomalies):")
    label_counts = classified_df['source_label'].value_counts()
    for label, count in label_counts.items():
        pct = (count / total) * 100
        print(f"  {label:15} : {count:>4,}  ({pct:5.1f}%)")

    # Train vs test split
    print("\nAnomaly counts by split:")
    for split_name, split_df in classified_df.groupby('split'):
        print(f"\n  [{split_name.upper()}] — {len(split_df):,} anomalies")
        split_counts = split_df['source_label'].value_counts()
        for label, count in split_counts.items():
            pct = (count / len(split_df)) * 100
            print(f"    {label:15} : {count:>4}  ({pct:5.1f}%)")

    # Per-city breakdown
    print("\nSource label counts by city:")
    city_label_pivot = (
        classified_df
        .groupby(['city', 'source_label'])
        .size()
        .unstack(fill_value=0)
    )
    print(city_label_pivot.to_string())

    # Distinct label types — checkpoint requirement
    distinct_labels = classified_df['source_label'].nunique()
    distinct_names  = sorted(classified_df['source_label'].unique())
    print(f"\n✓ Distinct source types: {distinct_labels}  — {distinct_names}")

    # Null check for test window
    test_rows  = classified_df[classified_df['split'] == 'test']
    null_count = test_rows['source_label'].isnull().sum()
    if null_count == 0:
        print(f"✓ All {len(test_rows):,} test-window anomalies have non-null source labels")
    else:
        print(f"✗ {null_count} test-window anomalies have null source labels — investigate")

    # Average confidence by label
    print("\nMean confidence by source label:")
    mean_conf = classified_df.groupby('source_label')['source_confidence'].mean()
    for label, conf in mean_conf.items():
        print(f"  {label:15} : {conf:.3f}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    print("\n" + "=" * 70)
    print("SmogAlert PK — Source Classifier (Stage 2)")
    print("SmogNet Datathon: Chemical Fingerprint Classification")
    print("=" * 70)

    # ------------------------------------------------------------------
    # STEP 1: Load cleaned training data for baseline statistics
    # ------------------------------------------------------------------
    print("\nSTEP 1: Loading cleaned dataset for training baselines...")

    if not os.path.exists(CLEANED_DATA_PATH):
        print(f"✗ Cleaned data not found at {CLEANED_DATA_PATH}")
        print("  Run: python src/preprocess.py first")
        return

    cleaned_df = pd.read_csv(CLEANED_DATA_PATH)
    print(f"  Loaded {len(cleaned_df):,} rows from {CLEANED_DATA_PATH}")

    # ------------------------------------------------------------------
    # STEP 2: Compute per-city-season training statistics
    # ------------------------------------------------------------------
    print("\nSTEP 2: Computing per-city-season training baselines...")
    training_stats = compute_training_stats(cleaned_df)

    # ------------------------------------------------------------------
    # STEP 3: Load anomalies (Stage 1 output)
    # ------------------------------------------------------------------
    print("\nSTEP 3: Loading anomalies from Stage 1...")

    if not os.path.exists(ANOMALIES_PATH):
        print(f"✗ Anomalies file not found at {ANOMALIES_PATH}")
        print("  Run: python src/model.py first")
        return

    anomalies_df = pd.read_csv(ANOMALIES_PATH)

    # Only keep rows that were flagged as anomalies (is_anomaly == 1)
    anomalies_df = anomalies_df[anomalies_df['is_anomaly'] == 1].copy()
    anomalies_df = anomalies_df.reset_index(drop=True)

    print(f"  Loaded {len(anomalies_df):,} anomaly rows")

    # Verify required columns are present
    required = ['pm25', 'pm10', 'so2', 'nh3', 'co', 'no', 'no2', 'city_season', 'split']
    missing  = [c for c in required if c not in anomalies_df.columns]
    if missing:
        print(f"✗ Missing columns in anomalies file: {missing}")
        return

    # ------------------------------------------------------------------
    # STEP 4: Classify each anomaly row
    # ------------------------------------------------------------------
    print("\nSTEP 4: Classifying anomalies by emission source...")
    print("  (This may take a moment for large files...)")

    classification_cols = anomalies_df.apply(
        lambda row: classify_row(row, training_stats),
        axis=1
    )

    # Merge the three new columns into the anomalies DataFrame
    anomalies_df['source_label']        = classification_cols['source_label']
    anomalies_df['source_confidence']   = classification_cols['source_confidence']
    anomalies_df['pollutant_signature'] = classification_cols['pollutant_signature']

    print(f"  ✓ Classification complete for {len(anomalies_df):,} rows")

    # ------------------------------------------------------------------
    # STEP 5: Print summary and validate checkpoint
    # ------------------------------------------------------------------
    print_classification_summary(anomalies_df)

    # ------------------------------------------------------------------
    # STEP 6: Save output
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 6: Saving classified anomalies...")

    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    anomalies_df.to_csv(OUTPUT_PATH, index=False)
    print(f"✓ Saved: {OUTPUT_PATH}  ({len(anomalies_df):,} rows)")

    # Checkpoint validation
    distinct = anomalies_df['source_label'].nunique()
    test_nulls = anomalies_df[anomalies_df['split'] == 'test']['source_label'].isnull().sum()

    print("\n" + "=" * 70)
    if distinct >= 3 and test_nulls == 0:
        print("✓ PHASE 3 CHECKPOINT PASSED")
        print(f"  — {distinct} distinct source types present")
        print(f"  — 0 null labels in test window")
        print("\nNext step: python src/alert_system.py")
    else:
        print("✗ PHASE 3 CHECKPOINT FAILED — review output above")
    print("=" * 70)


# ============================================================================
# Script Entry Point
# ============================================================================

if __name__ == "__main__":
    main()
