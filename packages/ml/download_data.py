"""
Data Loading Script for SmogAlert PK (SmogNet Datathon Version)
================================================================
This script loads the Pakistan Air Quality dataset (Kaggle) from local files.

The dataset is split into individual city files across two folders:
  - data/Training/ : Aug 2021 – Jul 2024 (historical data for model training)
  - data/Testing/  : Jul – Dec 2024     (recent data for model evaluation)

Each file has one city's hourly measurements of 8 pollutants:
  PM2.5, PM10, NO, NO2, SO2, NH3, CO, O3

This script merges all city files into one combined CSV for the pipeline.
"""

import pandas as pd   # For reading and merging data tables
import os             # For file path operations

# ============================================================================
# CONFIGURATION — File paths and city-to-filename mappings
# ============================================================================

# Root data directory (relative to project root)
DATA_DIR = "data"
TRAINING_DIR = os.path.join(DATA_DIR, "Training")
TESTING_DIR  = os.path.join(DATA_DIR, "Testing")

# Output file paths
# raw_aqi_data.csv  — used by all downstream scripts (preprocess, model, etc.)
# pakistan_aq_raw.csv — canonical merged copy kept for reference
RAW_OUTPUT    = os.path.join(DATA_DIR, "raw_aqi_data.csv")
MERGED_OUTPUT = os.path.join(DATA_DIR, "pakistan_aq_raw.csv")

# Mapping of city name → filename for training data
# Training files come in two formats: .xlsx (Islamabad/Karachi/Lahore)
# and .csv (Peshawar/Quetta)
TRAINING_FILES = {
    "Islamabad": "islamabad_complete_data.xlsx",
    "Karachi":   "karachi_complete_data.xlsx",
    "Lahore":    "lahore_complete_data.xlsx",
    "Peshawar":  "peshawar_complete_data.csv",
    "Quetta":    "quetta_complete_data.csv",
}

# Testing files are all CSV — one per city, covering Jul–Dec 2024
TESTING_FILES = {
    "Islamabad": "islamabad_complete_data_july_to_dec_2024.csv",
    "Karachi":   "karachi_complete_data_july_to_dec_2024.csv",
    "Lahore":    "lahore_complete_data_july_to_dec_2024.csv",
    "Peshawar":  "peshawar_complete_data_july_to_dec_2024.csv",
    "Quetta":    "quetta_complete_data_july_to_dec_2024.csv",
}

# The 8 pollutant columns required for the datathon pipeline
REQUIRED_POLLUTANT_COLUMNS = ['pm25', 'pm10', 'no', 'no2', 'so2', 'nh3', 'co', 'o3']

# ============================================================================
# FUNCTION 1: Normalize Column Names
# ============================================================================

def normalize_columns(df):
    """
    Standardize column names across all city files.

    Problem: Some files use dots in column names (e.g. "components.pm2_5")
    while others use underscores (e.g. "components_pm2_5"). We also want
    to strip the long "components_" prefix to get clean names like "pm25".

    Parameters:
        df (pandas.DataFrame): Raw DataFrame loaded from a city file

    Returns:
        pandas.DataFrame: Same data but with standardized column names
    """
    # Step 1: Replace dots with underscores in all column names
    # This makes "main.aqi" → "main_aqi" and "components.pm2_5" → "components_pm2_5"
    df.columns = [col.replace('.', '_') for col in df.columns]

    # Step 2: Rename the long component column names to short pollutant names
    # "components_pm2_5" → "pm25" (the underscore in pm2_5 becomes nothing, value stays pm25)
    rename_map = {
        'components_pm2_5': 'pm25',   # Fine particulate matter (most important for smog)
        'components_pm10':  'pm10',   # Coarse particulate matter
        'components_no':    'no',     # Nitric oxide (from combustion)
        'components_no2':   'no2',    # Nitrogen dioxide (traffic + industry)
        'components_so2':   'so2',    # Sulfur dioxide (industrial, fuel burning)
        'components_nh3':   'nh3',    # Ammonia (agriculture, crop burning)
        'components_co':    'co',     # Carbon monoxide (incomplete combustion)
        'components_o3':    'o3',     # Ozone (secondary pollutant from sunlight reactions)
        'main_aqi':         'aqi_raw' # Original AQI index from data source
    }

    # Apply only the renames that exist in this DataFrame
    # (different files may have different subsets of columns)
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    return df


# ============================================================================
# FUNCTION 2: Read a Single City File
# ============================================================================

def read_city_file(file_path, city_name):
    """
    Load one city's air quality file, standardize the date format, and tag
    each row with the city name.

    The dataset stores each city's data in a separate file without a 'city'
    column — we need to add that column ourselves based on which file we're
    reading.

    Date format problem: the files use inconsistent formats:
      - xlsx files  : "2021-08-24 00:00:00"   (YYYY-MM-DD — standard)
      - csv training: "24/08/2021 00:00:00"   (DD/MM/YYYY)
      - csv testing : "1/7/2024 0:00"         (D/M/YYYY)
    We parse all of them with dayfirst=True (treats ambiguous dates as DD/MM)
    and save back as ISO format "YYYY-MM-DD HH:MM:SS" so every city ends up
    with the same format before being concatenated.

    Parameters:
        file_path (str): Full path to the city data file (.csv or .xlsx)
        city_name (str): The city name to assign (e.g. "Lahore", "Karachi")

    Returns:
        pandas.DataFrame: City data with normalized column names and a 'city' column
    """
    # Choose the right reader based on file extension
    if file_path.endswith('.xlsx'):
        # Excel files need openpyxl library
        city_df = pd.read_excel(file_path)
    else:
        # Standard CSV files
        city_df = pd.read_csv(file_path)

    # Normalize column names (handle dot vs underscore, strip prefixes)
    city_df = normalize_columns(city_df)

    # Standardize the datetime column to ISO format (YYYY-MM-DD HH:MM:SS)
    # dayfirst=True tells pandas: when a date like "1/7/2024" is ambiguous,
    # treat the first number as the DAY (European convention), not the month.
    # This is correct for all Pakistani dataset files.
    if 'datetime' in city_df.columns:
        parsed_dates = pd.to_datetime(city_df['datetime'], dayfirst=True, errors='coerce')
        city_df['datetime'] = parsed_dates.dt.strftime('%Y-%m-%d %H:%M:%S')

    # Add city column — every row in this file belongs to city_name
    city_df['city'] = city_name

    return city_df


# ============================================================================
# FUNCTION 3: Load and Merge All City Files
# ============================================================================

def load_kaggle_dataset():
    """
    Read all Training and Testing city files and combine them into
    a single DataFrame.

    The combined DataFrame will have:
    - All 5 cities (Islamabad, Karachi, Lahore, Peshawar, Quetta)
    - Training rows: Aug 2021 – Jul 2024
    - Testing rows:  Jul 2024 – Dec 2024
    - 8 standardized pollutant columns
    - A 'city' column identifying each row's source

    Returns:
        pandas.DataFrame: All city data combined, or None if loading fails
    """
    all_city_frames = []  # We'll collect each city's DataFrame here

    # ------------------------------------------------------------------
    # Load Training Data (Aug 2021 – Jul 2024)
    # ------------------------------------------------------------------
    print("\nLoading training data (Aug 2021 – Jul 2024)...")
    print("-" * 50)

    for city_name, filename in TRAINING_FILES.items():
        file_path = os.path.join(TRAINING_DIR, filename)

        # Check the file exists before trying to load it
        if not os.path.exists(file_path):
            print(f"  ✗ File not found — skipping {city_name}: {file_path}")
            continue

        # Load the file and tag with city name
        city_df = read_city_file(file_path, city_name)
        print(f"  ✓ {city_name:12} — {len(city_df):>6,} rows loaded")
        all_city_frames.append(city_df)

    # ------------------------------------------------------------------
    # Load Testing Data (Jul – Dec 2024)
    # ------------------------------------------------------------------
    print("\nLoading testing data (Jul – Dec 2024)...")
    print("-" * 50)

    for city_name, filename in TESTING_FILES.items():
        file_path = os.path.join(TESTING_DIR, filename)

        if not os.path.exists(file_path):
            print(f"  ✗ File not found — skipping {city_name}: {file_path}")
            continue

        city_df = read_city_file(file_path, city_name)
        print(f"  ✓ {city_name:12} — {len(city_df):>6,} rows loaded")
        all_city_frames.append(city_df)

    # ------------------------------------------------------------------
    # Combine all city DataFrames into one
    # ------------------------------------------------------------------
    if not all_city_frames:
        print("✗ No data files were found. Cannot proceed.")
        return None

    # pd.concat stacks DataFrames vertically (row by row)
    # ignore_index=True resets the row numbers (0, 1, 2, ... N)
    combined_df = pd.concat(all_city_frames, ignore_index=True)

    print(f"\n✓ All cities combined: {len(combined_df):,} total rows")
    return combined_df


# ============================================================================
# FUNCTION 4: Validate Required Columns
# ============================================================================

def validate_columns(df):
    """
    Check that all required pollutant and metadata columns are present.

    This is a safety check — if a column is missing, the pipeline will
    fail later during preprocessing or model training, so we catch it here.

    Parameters:
        df (pandas.DataFrame): The merged dataset to validate

    Returns:
        bool: True if all required columns are present, False otherwise
    """
    required = REQUIRED_POLLUTANT_COLUMNS + ['city', 'datetime']
    missing_columns = [col for col in required if col not in df.columns]

    if missing_columns:
        print(f"\n✗ Validation failed — missing columns: {missing_columns}")
        print(f"  Available columns: {list(df.columns)}")
        return False

    print("✓ All required columns present")
    return True


# ============================================================================
# FUNCTION 5: Print Dataset Summary
# ============================================================================

def print_dataset_summary(df):
    """
    Print a human-readable overview of the loaded dataset.

    This helps verify that the data looks correct before running
    the full preprocessing pipeline.

    Parameters:
        df (pandas.DataFrame): The merged dataset to summarize
    """
    print("\n" + "=" * 70)
    print("DATASET SUMMARY")
    print("=" * 70)

    print(f"\nShape: {df.shape[0]:,} rows × {df.shape[1]} columns")

    # Show row count per city
    print("\nRows per city:")
    city_counts = df['city'].value_counts().sort_index()
    for city, count in city_counts.items():
        print(f"  {city:12} : {count:>6,} rows")

    # Show date range (use the raw datetime column before parsing)
    print(f"\nDate column sample values (raw):")
    print(f"  First row: {df['datetime'].iloc[0]}")
    print(f"  Last row:  {df['datetime'].iloc[-1]}")

    # Show missing value count per pollutant
    print("\nMissing values per pollutant:")
    for col in REQUIRED_POLLUTANT_COLUMNS:
        if col in df.columns:
            missing = df[col].isnull().sum()
            pct = (missing / len(df)) * 100
            print(f"  {col:6}: {missing:>5,} missing ({pct:.1f}%)")

    print("\nAll columns available in raw data:")
    for i, col in enumerate(df.columns, 1):
        print(f"  {i:2}. {col}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """
    Main function: load all city files, validate, summarize, and save.
    """
    print("\n" + "=" * 70)
    print("SmogAlert PK — Kaggle Dataset Loader")
    print("SmogNet Datathon: Pakistan Air Quality (5 Cities, 8 Pollutants)")
    print("=" * 70)

    # Step 1: Load all city files and merge
    print("\nSTEP 1: Loading city data files...")
    combined_df = load_kaggle_dataset()

    if combined_df is None:
        print("\n✗ Data loading failed. Check that Training/ and Testing/ folders exist.")
        return

    # Step 2: Validate required columns are present
    print("\nSTEP 2: Validating columns...")
    if not validate_columns(combined_df):
        print("✗ Cannot proceed — fix missing columns before running pipeline")
        return

    # Step 3: Print a summary of what was loaded
    print("\nSTEP 3: Dataset summary...")
    print_dataset_summary(combined_df)

    # Step 4: Save to both output files
    print("\n" + "=" * 70)
    print("STEP 4: Saving combined dataset...")
    print("=" * 70)

    combined_df.to_csv(MERGED_OUTPUT, index=False)
    print(f"✓ Saved canonical copy to: {MERGED_OUTPUT}")

    combined_df.to_csv(RAW_OUTPUT, index=False)
    print(f"✓ Saved pipeline input to:  {RAW_OUTPUT}")

    print("\n" + "=" * 70)
    print("✓ DATA LOADING COMPLETE!")
    print("=" * 70)
    print("\nNext step: python src/preprocess.py")


# ============================================================================
# Script Entry Point
# ============================================================================

if __name__ == "__main__":
    main()
