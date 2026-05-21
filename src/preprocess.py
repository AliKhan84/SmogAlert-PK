"""
Data Preprocessing Script for SmogAlert PK (SmogNet Datathon Version)
======================================================================
This script transforms raw city air quality data into a clean, feature-rich
dataset ready for machine learning.

Input : data/raw_aqi_data.csv  (merged 5-city, 8-pollutant dataset)
Output: data/cleaned_data.csv  (with time features, season, rolling averages,
                                AQI category, and train/test split labels)

Output schema:
    timestamp, city, pm25, pm10, no, no2, so2, nh3, co, o3,
    hour, day_of_week, month, season, is_weekend,
    pm25_24h_avg, pm10_24h_avg, aqi_category, split
"""

import pandas as pd    # For data manipulation
import numpy as np     # For numerical operations
import os              # For file path operations

# ============================================================================
# CONFIGURATION
# ============================================================================

INPUT_FILE  = "data/raw_aqi_data.csv"
OUTPUT_FILE = "data/cleaned_data.csv"

# Train/test split boundary:
# Rows before this date → 'train', rows on or after → 'test'
# This matches the Kaggle dataset structure (Testing/ folder = Jul–Dec 2024)
SPLIT_DATE = pd.Timestamp("2024-07-01")

# All 8 pollutant columns we need to validate and clean
POLLUTANT_COLUMNS = ['pm25', 'pm10', 'no', 'no2', 'so2', 'nh3', 'co', 'o3']

# Season definitions by month number
# Pakistan's smog season peaks in winter (Nov–Jan) due to crop burning + cold air trapping
SEASON_MAP = {
    1: 'Winter', 2: 'Winter',
    3: 'Spring', 4: 'Spring', 5: 'Spring',
    6: 'Summer', 7: 'Summer', 8: 'Summer', 9: 'Summer',
    10: 'Autumn',
    11: 'Winter', 12: 'Winter'
}

# ============================================================================
# FUNCTION 1: Load Raw Data
# ============================================================================

def load_data(file_path):
    """
    Load the merged raw air quality CSV file into a DataFrame.

    Parameters:
        file_path (str): Path to the raw CSV file

    Returns:
        pandas.DataFrame: The loaded raw data, or None if file not found
    """
    print("=" * 70)
    print("STEP 1: Loading Raw Data")
    print("=" * 70)

    if not os.path.exists(file_path):
        print(f"✗ File not found: {file_path}")
        print("  Run download_data.py first to generate this file.")
        return None

    df = pd.read_csv(file_path)
    print(f"✓ Loaded {len(df):,} rows from {file_path}")
    print(f"  Columns: {list(df.columns)}")
    return df


# ============================================================================
# FUNCTION 2: Parse and Standardize Timestamp
# ============================================================================

def parse_timestamp(df):
    """
    Convert the 'datetime' column to a proper Python datetime object
    and rename it to 'timestamp'.

    The raw files have mixed date formats across cities, for example:
      - "2021-08-24 00:00:00"  (YYYY-MM-DD format from xlsx files)
      - "24/08/2021 00:00:00"  (DD/MM/YYYY format from some CSVs)
      - "1/7/2024 0:00"        (M/D/YYYY H:MM from testing CSVs)

    pandas.to_datetime() with no format specified tries to parse all of these
    automatically. Rows that fail parsing become NaT (Not a Time) and are dropped.

    Parameters:
        df (pandas.DataFrame): Raw data with a 'datetime' string column

    Returns:
        pandas.DataFrame: Data with a proper 'timestamp' datetime column
    """
    print("\n" + "=" * 70)
    print("STEP 2: Parsing Timestamps")
    print("=" * 70)

    print(f"\nSample raw datetime values:")
    print(f"  Row 0: {df['datetime'].iloc[0]}")
    print(f"  Row -1: {df['datetime'].iloc[-1]}")

    # Convert string dates to Python datetime objects
    # errors='coerce' means: if a row can't be parsed, set it to NaT instead of crashing
    df['timestamp'] = pd.to_datetime(df['datetime'], errors='coerce')

    # Count rows where parsing failed (NaT = Not a Time)
    failed_parse_count = df['timestamp'].isnull().sum()
    if failed_parse_count > 0:
        print(f"⚠ Warning: {failed_parse_count} rows had unparseable dates — dropping them")
        df = df.dropna(subset=['timestamp'])
    else:
        print(f"✓ All {len(df):,} timestamps parsed successfully")

    # Drop the original string datetime column — we no longer need it
    df = df.drop(columns=['datetime'])

    # Sort rows by city and then by time so rolling averages work correctly later
    df = df.sort_values(['city', 'timestamp']).reset_index(drop=True)

    print(f"✓ Date range: {df['timestamp'].min()} → {df['timestamp'].max()}")
    return df


# ============================================================================
# FUNCTION 3: Filter Invalid Readings
# ============================================================================

def filter_invalid_readings(df):
    """
    Remove rows with physically impossible pollutant values.

    Sensor malfunctions can produce negative concentrations (e.g. PM2.5 = -200)
    which are impossible in reality. We drop any row where ANY pollutant
    column has a negative value.

    Parameters:
        df (pandas.DataFrame): Data that may contain invalid sensor readings

    Returns:
        pandas.DataFrame: Data with invalid rows removed
    """
    print("\n" + "=" * 70)
    print("STEP 3: Filtering Invalid Readings")
    print("=" * 70)

    rows_before = len(df)

    for pollutant in POLLUTANT_COLUMNS:
        if pollutant in df.columns:
            # Count how many negative values exist for this pollutant
            negative_count = (df[pollutant] < 0).sum()
            if negative_count > 0:
                print(f"  {pollutant:6}: removing {negative_count:,} rows with negative values")
            # Keep only rows where this pollutant value is 0 or positive
            df = df[df[pollutant] >= 0]

    rows_after  = len(df)
    rows_removed = rows_before - rows_after
    print(f"\n✓ Removed {rows_removed:,} invalid rows")
    print(f"  Rows remaining: {rows_after:,}")
    return df.reset_index(drop=True)


# ============================================================================
# FUNCTION 4: Handle Missing Values
# ============================================================================

def handle_missing_values(df):
    """
    Fill missing (NaN) pollutant values using forward-fill then backward-fill.

    Forward fill: if hour 3's PM2.5 is missing, use hour 2's value.
    Backward fill: if the very first hours of data are missing, use the
    first available future value.

    We do this separately per city so one city's data doesn't bleed into
    another city's gap.

    Parameters:
        df (pandas.DataFrame): Data that may have NaN values in pollutant columns

    Returns:
        pandas.DataFrame: Data with missing pollutant values filled in
    """
    print("\n" + "=" * 70)
    print("STEP 4: Handling Missing Values")
    print("=" * 70)

    total_filled = 0

    for pollutant in POLLUTANT_COLUMNS:
        if pollutant not in df.columns:
            continue

        missing_before = df[pollutant].isnull().sum()

        if missing_before > 0:
            # Fill within each city group separately
            # groupby('city') splits the DataFrame by city
            # transform applies ffill/bfill and returns results aligned to original index
            df[pollutant] = df.groupby('city')[pollutant].transform(
                lambda group: group.ffill().bfill()
            )

            missing_after = df[pollutant].isnull().sum()
            filled = missing_before - missing_after
            total_filled += filled
            print(f"  {pollutant:6}: filled {filled:,} values  ({missing_after} still missing)")
        else:
            print(f"  {pollutant:6}: no missing values")

    print(f"\n✓ Total values filled: {total_filled:,}")
    return df


# ============================================================================
# FUNCTION 5: Extract Time Features
# ============================================================================

def extract_time_features(df):
    """
    Create new columns that capture when each measurement was taken.

    Machine learning models cannot directly understand timestamps, but they
    can learn from numbers like hour=18 (rush hour) or month=12 (winter).
    These features help the model capture daily and seasonal pollution patterns.

    New columns added:
        hour       — 0 to 23 (morning vs evening rush hour)
        day_of_week— 0=Monday to 6=Sunday
        month      — 1 to 12 (January to December)
        is_weekend — 1 if Saturday/Sunday, 0 if weekday

    Parameters:
        df (pandas.DataFrame): Data with a parsed 'timestamp' column

    Returns:
        pandas.DataFrame: Data with 4 new time feature columns
    """
    print("\n" + "=" * 70)
    print("STEP 5: Extracting Time Features")
    print("=" * 70)

    # dt accessor lets us extract parts of a datetime column
    df['hour']        = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek   # 0 = Monday, 6 = Sunday
    df['month']       = df['timestamp'].dt.month

    # is_weekend: day_of_week 5 = Saturday, 6 = Sunday
    # Pollution patterns differ on weekends (less traffic, but sometimes more burning)
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)

    print(f"✓ Added: hour, day_of_week, month, is_weekend")
    print(f"\nSample of time features (first 3 rows):")
    print(df[['timestamp', 'hour', 'day_of_week', 'month', 'is_weekend']].head(3).to_string())
    return df


# ============================================================================
# FUNCTION 6: Add Season Column
# ============================================================================

def add_season(df):
    """
    Add a 'season' column based on the month of each measurement.

    Pakistan's pollution levels are strongly seasonal:
    - Winter (Nov–Feb): Temperature inversions trap smog near the ground.
                        Crop burning adds NH3 and CO. Visibility drops sharply.
    - Summer (Jun–Sep): Heat disperses pollutants upward. Less smog but high O3.
    - Spring (Mar–May): Moderate conditions, transitional.
    - Autumn (Oct):     Short transition before winter smog season begins.

    Season map used:
        Winter → months 11, 12, 1, 2
        Spring → months 3, 4, 5
        Summer → months 6, 7, 8, 9
        Autumn → month 10

    Parameters:
        df (pandas.DataFrame): Data with a 'month' column

    Returns:
        pandas.DataFrame: Data with new 'season' and 'city_season' columns
    """
    print("\n" + "=" * 70)
    print("STEP 6: Adding Season and City-Season Features")
    print("=" * 70)

    # Map each month number to its season name using the SEASON_MAP dictionary
    df['season'] = df['month'].map(SEASON_MAP)

    # city_season is a combined label like "Lahore_Winter" or "Karachi_Summer"
    # This lets the anomaly detector learn different normal ranges per city per season:
    # e.g. Lahore in Winter has very different "normal" PM2.5 vs Karachi in Summer
    df['city_season'] = df['city'] + "_" + df['season']

    print("✓ Added 'season' column")
    print("✓ Added 'city_season' column (e.g. 'Lahore_Winter')")

    # Show the distribution of rows across seasons
    print("\nSeason distribution:")
    season_counts = df['season'].value_counts()
    for season, count in season_counts.items():
        print(f"  {season:8}: {count:>7,} rows")

    print("\nCity-season groups:")
    city_season_counts = df['city_season'].value_counts().sort_index()
    for cs, count in city_season_counts.items():
        print(f"  {cs:25}: {count:>6,} rows")

    return df


# ============================================================================
# FUNCTION 7: Add Rolling Averages
# ============================================================================

def add_rolling_averages(df):
    """
    Compute 24-hour rolling averages for PM2.5 and PM10, separately per city.

    A rolling average looks at a sliding window of the past 24 hours and
    computes the mean. This smooths out hourly spikes and gives a sense of
    sustained pollution levels, which is what health guidelines use.

    Example: If PM2.5 at hour 10 is a spike due to a morning fire,
    the 24h rolling average will still reflect the day's overall trend.

    We compute rolling averages per-city because mixing Lahore data with
    Karachi data in the same window would produce meaningless averages.

    Parameters:
        df (pandas.DataFrame): Data sorted by city+timestamp with pm25 and pm10 columns

    Returns:
        pandas.DataFrame: Data with 'pm25_24h_avg' and 'pm10_24h_avg' columns added
    """
    print("\n" + "=" * 70)
    print("STEP 7: Computing 24-Hour Rolling Averages")
    print("=" * 70)

    # rolling(24) computes the mean of the current row + 23 previous rows
    # min_periods=1 means: even for the first few rows with fewer than 24 hours of history,
    # compute the average of however many rows we do have
    # transform() aligns the result back to the original row index after groupby

    df['pm25_24h_avg'] = df.groupby('city')['pm25'].transform(
        lambda group: group.rolling(window=24, min_periods=1).mean()
    )

    df['pm10_24h_avg'] = df.groupby('city')['pm10'].transform(
        lambda group: group.rolling(window=24, min_periods=1).mean()
    )

    print("✓ Added 'pm25_24h_avg' (24-hour rolling mean of PM2.5, per city)")
    print("✓ Added 'pm10_24h_avg' (24-hour rolling mean of PM10, per city)")

    # Sanity check: rolling average should be smoother (lower std) than raw values
    print(f"\nPM2.5  — Raw Mean: {df['pm25'].mean():.1f},  "
          f"Raw Std: {df['pm25'].std():.1f}")
    print(f"PM2.5  — Avg Mean: {df['pm25_24h_avg'].mean():.1f},  "
          f"Avg Std: {df['pm25_24h_avg'].std():.1f}  (should be lower)")

    return df


# ============================================================================
# FUNCTION 8: Categorize AQI
# ============================================================================

def categorize_aqi(df):
    """
    Convert the continuous PM2.5 value into a 4-tier AQI category label.

    These categories are based on international PM2.5 health thresholds
    and are used to color-code the map and trigger alerts:

        Good       (PM2.5 ≤ 50 µg/m³) : Air quality satisfactory
        Moderate   (51–100 µg/m³)      : Acceptable; sensitive groups may be affected
        Unhealthy  (101–150 µg/m³)     : Everyone may experience health effects
        Hazardous  (> 150 µg/m³)       : Emergency conditions — serious health risk

    Alerts are only generated for Unhealthy and Hazardous levels.

    Parameters:
        df (pandas.DataFrame): Data with a cleaned 'pm25' column

    Returns:
        pandas.DataFrame: Data with new 'aqi_category' column
    """
    print("\n" + "=" * 70)
    print("STEP 8: Categorizing AQI Levels")
    print("=" * 70)

    def assign_aqi_category(pm25_value):
        """
        Map a single PM2.5 float value to its AQI category string.

        Parameters:
            pm25_value (float): PM2.5 concentration in µg/m³

        Returns:
            str: One of 'Good', 'Moderate', 'Unhealthy', 'Hazardous'
        """
        if pm25_value <= 50:
            return "Good"
        elif pm25_value <= 100:
            return "Moderate"
        elif pm25_value <= 150:
            return "Unhealthy"
        else:
            return "Hazardous"

    # Apply the function to every row in the pm25 column
    df['aqi_category'] = df['pm25'].apply(assign_aqi_category)

    print("✓ Added 'aqi_category' column")
    print("\nAQI level distribution:")
    aqi_counts = df['aqi_category'].value_counts()
    for level in ["Good", "Moderate", "Unhealthy", "Hazardous"]:
        if level in aqi_counts:
            count = aqi_counts[level]
            pct   = (count / len(df)) * 100
            bar   = "█" * int(pct / 2)
            print(f"  {level:10} : {count:>7,} rows ({pct:5.1f}%) {bar}")

    return df


# ============================================================================
# FUNCTION 9: Add Train/Test Split Column
# ============================================================================

def add_split_column(df):
    """
    Label each row as either 'train' or 'test' based on its timestamp.

    The boundary is 2024-07-01 (matching how the Kaggle dataset is organized):
      - Before 2024-07-01  → 'train'  (historical data for model training)
      - 2024-07-01 onward  → 'test'   (held-out data for model evaluation)

    This is important for fair evaluation: the model must never train on
    data from the same period it's tested on, otherwise results are misleading.

    Parameters:
        df (pandas.DataFrame): Data with a 'timestamp' datetime column

    Returns:
        pandas.DataFrame: Data with a new 'split' column ('train' or 'test')
    """
    print("\n" + "=" * 70)
    print("STEP 9: Adding Train/Test Split Labels")
    print("=" * 70)

    # np.where(condition, value_if_true, value_if_false) is like an IF formula in Excel
    df['split'] = np.where(df['timestamp'] < SPLIT_DATE, 'train', 'test')

    train_count = (df['split'] == 'train').sum()
    test_count  = (df['split'] == 'test').sum()

    print(f"✓ Added 'split' column (boundary: {SPLIT_DATE.date()})")
    print(f"  Train rows: {train_count:>7,}  (before {SPLIT_DATE.date()})")
    print(f"  Test rows:  {test_count:>7,}  ({SPLIT_DATE.date()} – Dec 2024)")

    # Verify test window exists
    if test_count == 0:
        print("⚠ Warning: No test rows found! Check timestamp parsing.")

    return df


# ============================================================================
# FUNCTION 10: Select and Order Output Columns
# ============================================================================

def select_output_columns(df):
    """
    Keep only the columns defined in the output schema and drop extras.

    The output schema (defined in EXECUTION_PLAN.md Phase 1) is:
        timestamp, city, pm25, pm10, no, no2, so2, nh3, co, o3,
        hour, day_of_week, month, season, is_weekend,
        pm25_24h_avg, pm10_24h_avg, aqi_category, split

    We also keep 'city_season' because Phase 2 (anomaly detection)
    needs it to group rows for per-group Isolation Forest training.

    Parameters:
        df (pandas.DataFrame): Fully processed DataFrame with all columns

    Returns:
        pandas.DataFrame: DataFrame with only the required output columns
    """
    output_columns = [
        'timestamp', 'city',
        'pm25', 'pm10', 'no', 'no2', 'so2', 'nh3', 'co', 'o3',
        'hour', 'day_of_week', 'month', 'season', 'city_season', 'is_weekend',
        'pm25_24h_avg', 'pm10_24h_avg',
        'aqi_category', 'split'
    ]

    # Keep only columns that actually exist (safety check)
    available_output_cols = [col for col in output_columns if col in df.columns]
    missing_output_cols   = [col for col in output_columns if col not in df.columns]

    if missing_output_cols:
        print(f"⚠ Warning: These expected columns are missing: {missing_output_cols}")

    return df[available_output_cols]


# ============================================================================
# FUNCTION 11: Save Cleaned Data
# ============================================================================

def save_cleaned_data(df, output_path):
    """
    Save the preprocessed DataFrame to a CSV file.

    Parameters:
        df (pandas.DataFrame): The fully cleaned and feature-engineered DataFrame
        output_path (str): Path where the CSV should be saved
    """
    print("\n" + "=" * 70)
    print("STEP 10: Saving Cleaned Data")
    print("=" * 70)

    # Make sure the output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    df.to_csv(output_path, index=False)

    print(f"✓ Saved to: {output_path}")
    print(f"  Rows:    {len(df):,}")
    print(f"  Columns: {len(df.columns)}")
    file_size_kb = os.path.getsize(output_path) / 1024
    print(f"  Size:    {file_size_kb:.1f} KB")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """
    Run all preprocessing steps in order to transform raw data into
    a clean, feature-rich dataset ready for machine learning.
    """
    print("\n" + "=" * 70)
    print("SmogAlert PK — Data Preprocessing Pipeline")
    print("SmogNet Datathon: 8-Pollutant, 5-City Dataset")
    print("=" * 70)

    # Step 1: Load raw merged data
    df = load_data(INPUT_FILE)
    if df is None:
        print("\n✗ Preprocessing failed: cannot load input data")
        return

    # Step 2: Parse timestamps and sort by city+time
    df = parse_timestamp(df)

    # Step 3: Drop physically impossible negative readings
    df = filter_invalid_readings(df)

    # Step 4: Fill missing pollutant values per city (ffill then bfill)
    df = handle_missing_values(df)

    # Step 5: Extract time-of-day and weekday features
    df = extract_time_features(df)

    # Step 6: Add season and city_season composite features
    df = add_season(df)

    # Step 7: Compute 24-hour rolling averages (per city)
    df = add_rolling_averages(df)

    # Step 8: Assign 4-tier AQI category labels
    df = categorize_aqi(df)

    # Step 9: Label each row as train or test
    df = add_split_column(df)

    # Step 10: Keep only the required output columns
    df = select_output_columns(df)

    # Step 11: Save to CSV
    save_cleaned_data(df, OUTPUT_FILE)

    # Final summary
    print("\n" + "=" * 70)
    print("✓ PREPROCESSING COMPLETE!")
    print("=" * 70)
    print(f"\nOutput file: {OUTPUT_FILE}")
    print(f"\nFinal column list:")
    for i, col in enumerate(df.columns, 1):
        print(f"  {i:2}. {col}")

    # Verify Phase 1 completion checkpoint
    print("\n--- Phase 1 Checkpoint Verification ---")
    cities_present = sorted(df['city'].unique())
    print(f"Cities:     {cities_present}")

    pollutants_present = [c for c in POLLUTANT_COLUMNS if c in df.columns]
    print(f"Pollutants: {pollutants_present}")

    test_rows = (df['split'] == 'test').sum()
    print(f"Test rows (Jul–Dec 2024): {test_rows:,}")

    has_season    = 'season' in df.columns
    has_city_seas = 'city_season' in df.columns
    has_split     = 'split' in df.columns
    print(f"season column present:      {has_season}")
    print(f"city_season column present: {has_city_seas}")
    print(f"split column present:       {has_split}")

    if all([len(cities_present) > 1, len(pollutants_present) == 8,
            test_rows > 0, has_season, has_split]):
        print("\n✓ Phase 1 checkpoint PASSED — ready for Phase 2 (anomaly detection)")
    else:
        print("\n✗ Phase 1 checkpoint FAILED — review warnings above")

    print("\nNext step: python src/model.py")


# ============================================================================
# Script Entry Point
# ============================================================================

if __name__ == "__main__":
    main()
