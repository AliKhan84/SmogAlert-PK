"""
Enhanced Alert Generation — SmogAlert PK (SmogNet Datathon)
===========================================================
Stage 3 of the 3-stage pipeline: generate structured, public-facing
air quality alerts driven by the source classification output from Stage 2.

How it works:
  1. Load outputs/anomalies_classified.csv (Stage 2 output)
  2. Filter to: test split (Jul–Dec 2024) AND AQI level Unhealthy or Hazardous
  3. For each qualifying anomaly, apply a source-specific alert template
  4. Save outputs/alerts_log.csv with full bilingual structured alert schema

Why template-based?
  Each emission source causes different health risks and calls for different
  protective actions. A crop-burning alert should mention ammonia and crop
  residue; a dust-storm alert should mention coarse particulates. Generic
  colour-level messages don't give the public actionable information.

Output schema:
  timestamp, city, aqi_level, source_label, pollutant_signature,
  affected_groups, protective_actions, alert_text_en, alert_text_ur, split
"""

import os
import pandas as pd
import numpy as np

# ============================================================================
# CONFIGURATION
# ============================================================================

OUTPUTS_DIR            = "outputs"
CLASSIFIED_PATH        = os.path.join(OUTPUTS_DIR, "anomalies_classified.csv")
ALERTS_LOG_PATH        = os.path.join(OUTPUTS_DIR, "alerts_log.csv")

# Only generate alerts for anomalies in the test window (Jul–Dec 2024)
TARGET_SPLIT = "test"

# Only alert for readings above this AQI tier
ALERT_AQI_LEVELS = {"Unhealthy", "Hazardous"}

# If a city has no Unhealthy/Hazardous test anomalies, fall back to Moderate
FALLBACK_AQI_LEVEL = "Moderate"


# ============================================================================
# TEMPLATE LIBRARY — one entry per source type
#
# Each entry has:
#   alert_text_en    — English alert paragraph ([CITY] is replaced at runtime)
#   alert_text_ur    — Urdu translation (same structure)
#   affected_groups  — Who is at highest risk
#   protective_actions — What people should do
# ============================================================================

ALERT_TEMPLATES = {

    "crop_burning": {
        "alert_text_en": (
            "⚠ Air Quality Alert — [CITY]. "
            "Elevated levels of ammonia and carbon monoxide indicate active crop burning "
            "in or near the area, causing hazardous smog conditions. "
            "Children, elderly residents, and individuals with asthma or respiratory "
            "conditions are at highest risk. "
            "Avoid outdoor activity, keep windows closed, and use an N95 mask if you "
            "must go outside."
        ),
        "alert_text_ur": (
            "⚠ فضائی آلودگی الرٹ — [CITY]۔ "
            "امونیا اور کاربن مونو آکسائیڈ کی بلند سطح سے پتہ چلتا ہے کہ قریبی علاقوں میں "
            "فصل کی باقیات جلائی جا رہی ہے جس سے خطرناک دھند پیدا ہو رہی ہے۔ "
            "بچے، بزرگ اور سانس کی تکلیف والے افراد سب سے زیادہ خطرے میں ہیں۔ "
            "باہر جانے سے گریز کریں، کھڑکیاں بند رکھیں اور باہر نکلنا ضروری ہو تو N95 ماسک پہنیں۔"
        ),
        "affected_groups":    "Children, elderly, asthma and respiratory patients",
        "protective_actions": "Avoid outdoor activity; keep windows closed; wear N95 mask if going outside",
    },

    "vehicular": {
        "alert_text_en": (
            "⚠ Air Quality Alert — [CITY]. "
            "High nitrogen oxide levels suggest heavy vehicular traffic is contributing to "
            "dangerous air pollution. "
            "Children, the elderly, and those with lung or heart conditions should limit "
            "outdoor exposure. "
            "Avoid high-traffic areas, use public transport if possible, and wear a mask outdoors."
        ),
        "alert_text_ur": (
            "⚠ فضائی آلودگی الرٹ — [CITY]۔ "
            "نائٹروجن آکسائیڈ کی بلند سطح سے ظاہر ہوتا ہے کہ بھاری ٹریفک فضائی آلودگی کا سبب بن رہی ہے۔ "
            "بچے، بزرگ اور پھیپھڑوں یا دل کی تکلیف والے افراد باہر کم سے کم نکلیں۔ "
            "بھاری ٹریفک والے علاقوں سے بچیں، ممکن ہو تو پبلک ٹرانسپورٹ استعمال کریں اور باہر ماسک پہنیں۔"
        ),
        "affected_groups":    "Children, elderly, patients with lung or heart conditions",
        "protective_actions": "Avoid high-traffic areas; use public transport; wear mask outdoors",
    },

    "industrial": {
        "alert_text_en": (
            "⚠ Air Quality Alert — [CITY]. "
            "Elevated sulfur dioxide readings point to industrial emissions as a probable "
            "cause of the current pollution spike. "
            "Respiratory patients, children, and elderly individuals are especially vulnerable. "
            "Stay indoors, close ventilation, and seek medical attention if experiencing "
            "breathing difficulty."
        ),
        "alert_text_ur": (
            "⚠ فضائی آلودگی الرٹ — [CITY]۔ "
            "سلفر ڈائی آکسائیڈ کی بلند مقدار سے لگتا ہے کہ صنعتی اخراج آلودگی کی موجودہ لہر کا ممکنہ سبب ہے۔ "
            "سانس کے مریض، بچے اور بزرگ خاص طور پر متاثر ہو سکتے ہیں۔ "
            "گھر کے اندر رہیں، وینٹیلیشن بند کریں اور سانس لینے میں دشواری ہو تو فوری طبی مدد لیں۔"
        ),
        "affected_groups":    "Respiratory patients, children, elderly",
        "protective_actions": "Stay indoors; close ventilation; seek medical attention if breathing difficulty",
    },

    "dust_storm": {
        "alert_text_en": (
            "⚠ Air Quality Alert — [CITY]. "
            "A high ratio of coarse to fine particulates suggests an active dust storm is "
            "reducing air quality. "
            "People with respiratory conditions, children, and the elderly should remain "
            "indoors with windows and doors sealed. "
            "Avoid all outdoor activity until conditions improve."
        ),
        "alert_text_ur": (
            "⚠ فضائی آلودگی الرٹ — [CITY]۔ "
            "موٹے اور باریک ذرات کا زیادہ تناسب ظاہر کرتا ہے کہ طوفان بادِ خاک سے فضا خراب ہو رہی ہے۔ "
            "سانس کے مریض، بچے اور بزرگ کھڑکیاں اور دروازے بند کر کے گھر کے اندر رہیں۔ "
            "حالات بہتر ہونے تک ہر قسم کی بیرونی سرگرمی سے گریز کریں۔"
        ),
        "affected_groups":    "People with respiratory conditions, children, elderly",
        "protective_actions": "Stay indoors; seal windows and doors; avoid all outdoor activity",
    },

    "general_smog": {
        "alert_text_en": (
            "⚠ Air Quality Alert — [CITY]. "
            "Dangerously high concentrations of fine particulate matter (PM2.5) have been "
            "detected, consistent with dense urban smog from diffuse low-level sources such "
            "as traffic, domestic heating, and local industry acting together. "
            "Children, elderly residents, and people with respiratory or heart conditions are "
            "at serious risk. "
            "Remain indoors, keep windows closed, and use an air purifier if available."
        ),
        "alert_text_ur": (
            "⚠ فضائی آلودگی الرٹ — [CITY]۔ "
            "باریک ذرات (PM2.5) کی انتہائی خطرناک سطح ریکارڈ کی گئی ہے جو گھنے شہری دھوئیں کی علامت ہے — "
            "ٹریفک، گھریلو حرارت اور مقامی صنعت مل کر اس کا سبب بن رہے ہیں۔ "
            "بچے، بزرگ اور دل و پھیپھڑوں کے مریض شدید خطرے میں ہیں۔ "
            "گھر کے اندر رہیں، کھڑکیاں بند رکھیں اور ایئر پیوریفائر استعمال کریں اگر دستیاب ہو۔"
        ),
        "affected_groups":    "Children, elderly, people with respiratory or heart conditions",
        "protective_actions": "Remain indoors; keep windows closed; use air purifier if available",
    },

    # mixed and unclassified share a general template
    "mixed": {
        "alert_text_en": (
            "⚠ Air Quality Alert — [CITY]. "
            "Pollution levels have spiked above safe thresholds due to a combination of "
            "emission sources in the area. "
            "All sensitive groups — including children, elderly, and those with respiratory "
            "conditions — should minimise outdoor activity and wear protective masks."
        ),
        "alert_text_ur": (
            "⚠ فضائی آلودگی الرٹ — [CITY]۔ "
            "آلودگی کی سطح کئی ذرائع کے امتزاج سے محفوظ حد سے بڑھ گئی ہے۔ "
            "تمام حساس افراد — بشمول بچے، بزرگ اور سانس کی تکلیف والے — "
            "باہر کا وقت کم سے کم رکھیں اور حفاظتی ماسک پہنیں۔"
        ),
        "affected_groups":    "Children, elderly, people with respiratory or heart conditions",
        "protective_actions": "Minimise outdoor activity; wear N95 or surgical mask outdoors",
    },

    "unclassified": {
        "alert_text_en": (
            "⚠ Air Quality Alert — [CITY]. "
            "Pollution levels have spiked above safe thresholds due to a combination of "
            "emission sources in the area. "
            "All sensitive groups — including children, elderly, and those with respiratory "
            "conditions — should minimise outdoor activity and wear protective masks."
        ),
        "alert_text_ur": (
            "⚠ فضائی آلودگی الرٹ — [CITY]۔ "
            "آلودگی کی سطح محفوظ حد سے بڑھ گئی ہے۔ "
            "تمام حساس افراد — بشمول بچے، بزرگ اور سانس کی تکلیف والے — "
            "باہر کا وقت کم سے کم رکھیں اور حفاظتی ماسک پہنیں۔"
        ),
        "affected_groups":    "Children, elderly, people with respiratory or heart conditions",
        "protective_actions": "Minimise outdoor activity; wear N95 or surgical mask outdoors",
    },
}


# ============================================================================
# FUNCTION 1: Load Classified Anomalies
# ============================================================================

def load_classified_anomalies():
    """
    Load the classified anomaly file produced by Stage 2 (source_classifier.py).

    Parameters: none

    Returns:
        pd.DataFrame or None: The classified anomaly records, or None on failure.
    """
    if not os.path.exists(CLASSIFIED_PATH):
        print(f"✗ Classified anomalies not found at {CLASSIFIED_PATH}")
        print("  Run: python src/source_classifier.py first")
        return None

    df = pd.read_csv(CLASSIFIED_PATH)
    print(f"  Loaded {len(df):,} classified anomaly rows from {CLASSIFIED_PATH}")
    return df


# ============================================================================
# FUNCTION 2: Select Rows That Qualify for an Alert
# ============================================================================

def select_alert_candidates(df):
    """
    Filter the classified anomaly DataFrame to rows that warrant a public alert:
      - Must be in the test split (Jul–Dec 2024 — the held-out evaluation window)
      - Must have AQI level Unhealthy or Hazardous

    If a city has no Unhealthy/Hazardous test anomalies, the city's Moderate
    test anomalies are included so every city is represented in the output.

    Parameters:
        df (pd.DataFrame): Full classified anomaly DataFrame.

    Returns:
        pd.DataFrame: Subset of rows that qualify for alert generation.
    """
    # Step 1: keep only test-window rows
    test_df = df[df['split'] == TARGET_SPLIT].copy()
    print(f"  Test-window anomalies: {len(test_df):,}")

    # Step 2: keep Unhealthy/Hazardous
    primary_df = test_df[test_df['aqi_category'].isin(ALERT_AQI_LEVELS)].copy()
    print(f"  Qualifying (Unhealthy/Hazardous): {len(primary_df):,}")

    # Step 3: for any city absent from primary_df, pull in Moderate as fallback
    cities_with_alerts = set(primary_df['city'].unique())
    all_test_cities    = set(test_df['city'].unique())
    missing_cities     = all_test_cities - cities_with_alerts

    fallback_rows = []
    for city in missing_cities:
        city_fallback = test_df[
            (test_df['city'] == city) &
            (test_df['aqi_category'] == FALLBACK_AQI_LEVEL)
        ]
        if len(city_fallback) > 0:
            fallback_rows.append(city_fallback)
            print(f"  Fallback to Moderate for {city}: {len(city_fallback)} rows added")

    if fallback_rows:
        primary_df = pd.concat([primary_df] + fallback_rows, ignore_index=True)

    return primary_df


# ============================================================================
# FUNCTION 3: Generate One Alert Record
# ============================================================================

def generate_alert_record(row):
    """
    Build a single structured alert record for one anomaly row.

    Uses the source_label to look up the right template and fills in the
    city name and pollutant signature at runtime.

    Parameters:
        row (pd.Series): One row from the qualified anomaly DataFrame.
                         Must have: timestamp, city, aqi_category, source_label,
                                    pollutant_signature, split.

    Returns:
        dict: One alert record matching the output schema.
    """
    city          = row['city']
    source_label  = row.get('source_label', 'unclassified')
    signature     = row.get('pollutant_signature', '—')

    # Look up template; fall back to 'unclassified' if label not found
    template = ALERT_TEMPLATES.get(source_label, ALERT_TEMPLATES['unclassified'])

    # Insert the city name into the template placeholders
    alert_en = template['alert_text_en'].replace('[CITY]', city)
    alert_ur = template['alert_text_ur'].replace('[CITY]', city)

    return {
        'timestamp':           row['timestamp'],
        'city':                city,
        'aqi_level':           row['aqi_category'],
        'source_label':        source_label,
        'pollutant_signature': signature,
        'affected_groups':     template['affected_groups'],
        'protective_actions':  template['protective_actions'],
        'alert_text_en':       alert_en,
        'alert_text_ur':       alert_ur,
        'split':               row['split'],
    }


# ============================================================================
# FUNCTION 4: Generate All Alerts
# ============================================================================

def generate_all_alerts(candidates_df):
    """
    Apply generate_alert_record() to every qualifying row and return
    the full alerts DataFrame.

    Parameters:
        candidates_df (pd.DataFrame): Output of select_alert_candidates().

    Returns:
        pd.DataFrame: All generated alerts in output schema order.
    """
    alert_records = [generate_alert_record(row) for _, row in candidates_df.iterrows()]
    alerts_df = pd.DataFrame(alert_records)

    # Sort by city then timestamp so the log is easy to browse
    alerts_df = alerts_df.sort_values(['city', 'timestamp']).reset_index(drop=True)

    return alerts_df


# ============================================================================
# FUNCTION 5: Print Alert Summary
# ============================================================================

def print_alert_summary(alerts_df):
    """
    Print a formatted summary of the generated alerts.

    Parameters:
        alerts_df (pd.DataFrame): Generated alerts output DataFrame.
    """
    print("\n" + "=" * 70)
    print("ALERT SUMMARY")
    print("=" * 70)

    total = len(alerts_df)
    print(f"\nTotal alerts generated: {total:,}")

    # Per-city count
    print("\nAlerts per city:")
    city_counts = alerts_df['city'].value_counts().sort_index()
    for city, count in city_counts.items():
        print(f"  {city:12} : {count:>4,}")

    # Source label distribution
    print("\nAlerts by source label:")
    label_counts = alerts_df['source_label'].value_counts()
    for label, count in label_counts.items():
        pct = (count / total) * 100
        print(f"  {label:15} : {count:>4,}  ({pct:5.1f}%)")

    # AQI level distribution
    print("\nAlerts by AQI level:")
    aqi_counts = alerts_df['aqi_level'].value_counts()
    for level, count in aqi_counts.items():
        pct = (count / total) * 100
        print(f"  {level:12} : {count:>4,}  ({pct:5.1f}%)")

    # Print sample alerts (one per source type for readability)
    print("\nSAMPLE ALERTS (one per source type):")
    print("-" * 70)

    shown_labels = set()
    for _, row in alerts_df.iterrows():
        label = row['source_label']
        if label in shown_labels:
            continue
        shown_labels.add(label)

        print(f"\n[{row['timestamp']}]  {row['city']}  |  {row['aqi_level']}  |  {label}")
        print(f"  Signature   : {row['pollutant_signature']}")
        print(f"  At risk     : {row['affected_groups']}")
        print(f"  Actions     : {row['protective_actions']}")
        print(f"  English     : {row['alert_text_en'][:120]}...")
        print(f"  Urdu        : {row['alert_text_ur'][:80]}...")

        if len(shown_labels) >= 6:
            break

    # Checkpoint check
    distinct_sources = alerts_df['source_label'].nunique()
    has_required_cols = all(
        c in alerts_df.columns
        for c in ['source_label', 'affected_groups', 'protective_actions',
                  'alert_text_en', 'alert_text_ur']
    )
    null_source = alerts_df['source_label'].isnull().sum()

    print("\n" + "=" * 70)
    if distinct_sources >= 3 and has_required_cols and null_source == 0:
        print("✓ PHASE 4 CHECKPOINT PASSED")
        print(f"  — {distinct_sources} distinct source types in alerts")
        print(f"  — All required columns present and non-null")
    else:
        print("✗ PHASE 4 CHECKPOINT FAILED — review output above")
    print("=" * 70)


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    print("\n" + "=" * 70)
    print("SmogAlert PK — Enhanced Alert Generation (Stage 3)")
    print("SmogNet Datathon: Source-Driven Bilingual Public Alerts")
    print("=" * 70)

    # ------------------------------------------------------------------
    # STEP 1: Load Stage 2 output
    # ------------------------------------------------------------------
    print("\nSTEP 1: Loading classified anomalies...")
    classified_df = load_classified_anomalies()
    if classified_df is None:
        return

    # ------------------------------------------------------------------
    # STEP 2: Select rows that qualify for alerts
    # ------------------------------------------------------------------
    print("\nSTEP 2: Selecting alert candidates...")
    candidates_df = select_alert_candidates(classified_df)

    if len(candidates_df) == 0:
        print("✗ No qualifying anomalies found. Cannot generate alerts.")
        return

    print(f"\n  ✓ {len(candidates_df):,} rows selected for alert generation")

    # ------------------------------------------------------------------
    # STEP 3: Generate alerts
    # ------------------------------------------------------------------
    print("\nSTEP 3: Applying alert templates...")
    alerts_df = generate_all_alerts(candidates_df)
    print(f"  ✓ Generated {len(alerts_df):,} structured alerts")

    # ------------------------------------------------------------------
    # STEP 4: Save output
    # ------------------------------------------------------------------
    print("\nSTEP 4: Saving alerts log...")
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    alerts_df.to_csv(ALERTS_LOG_PATH, index=False)
    print(f"  ✓ Saved: {ALERTS_LOG_PATH}  ({len(alerts_df):,} rows)")

    # ------------------------------------------------------------------
    # STEP 5: Summary + checkpoint
    # ------------------------------------------------------------------
    print_alert_summary(alerts_df)

    print("\nNext step: python dashboard/app.py  (Phase 5)")


# ============================================================================
# Script Entry Point
# ============================================================================

if __name__ == "__main__":
    main()
