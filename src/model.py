"""
Air Quality Prediction Model Training Script for SmogAlert PK

This script trains a Random Forest Classifier to predict AQI danger levels
based on PM2.5 readings and time-based features.

WHAT IS A RANDOM FOREST?
========================
Imagine you're trying to decide if the air quality is dangerous. Instead of
asking just one expert, you ask 100 different experts (trees), and each one
gives their opinion based on different aspects of the data. Then you take a
vote - whatever the majority says is your final answer.

That's exactly what a Random Forest does:
- It creates many decision trees (a "forest" of trees)
- Each tree learns from a random subset of the data
- Each tree makes a prediction
- The final prediction is the majority vote from all trees

Why is this better than one tree?
- More robust: If one tree makes a mistake, others can correct it
- Less overfitting: Random sampling prevents memorizing the training data
- More accurate: Combining many weak learners creates a strong learner

In our case, the Random Forest will learn patterns like:
"If PM2.5 > 150 AND hour is 8am AND it's winter, then AQI = Hazardous"
"""

# ============================================================================
# IMPORT LIBRARIES
# ============================================================================

import pandas as pd              # For data manipulation
import numpy as np               # For numerical operations
import matplotlib.pyplot as plt  # For creating plots and charts
import seaborn as sns           # For beautiful statistical visualizations
import joblib                   # For saving/loading machine learning models
import os                       # For file and directory operations

# Facebook Prophet for time-series forecasting
# Prophet requires 'ds' (datestamp) and 'y' (value) column names — explained in forecast function
from prophet import Prophet

# Machine learning libraries from scikit-learn
from sklearn.model_selection import train_test_split  # For splitting data into train/test sets
from sklearn.ensemble import RandomForestClassifier   # The Random Forest algorithm
from sklearn.ensemble import IsolationForest          # For anomaly detection
from sklearn.metrics import (
    accuracy_score,           # Measures overall correctness
    classification_report,    # Detailed metrics per class
    confusion_matrix         # Shows where the model gets confused
)

# ============================================================================
# CONFIGURATION
# ============================================================================

# File paths - PART 1: Random Forest
INPUT_FILE = "data/cleaned_data.csv"
MODEL_OUTPUT = "models/random_forest_model.pkl"
CONFUSION_MATRIX_OUTPUT = "outputs/confusion_matrix.png"
FEATURE_IMPORTANCE_OUTPUT = "outputs/feature_importance.png"

# File paths - PART 2: Isolation Forest (Anomaly Detection)
# Each city-season group gets its own model file, e.g. models/isolation_forest_Lahore_Winter.pkl
ISOLATION_FOREST_DIR = "models"
ANOMALY_PLOT_OUTPUT = "outputs/anomaly_plot.png"
ANOMALIES_OUTPUT = "outputs/anomalies.csv"
# Minimum rows needed in a city-season group to train a meaningful model
MIN_ROWS_PER_GROUP = 100

# Model parameters - PART 1: Random Forest
RANDOM_STATE = 42  # We'll explain this below
N_ESTIMATORS = 100  # Number of trees in the forest
TEST_SIZE = 0.2     # 20% of data for testing, 80% for training

# Model parameters - PART 2: Isolation Forest
CONTAMINATION = 0.05  # Expect 5% of data to be anomalies

# File paths - PART 3: Prophet Forecasting
FORECAST_PLOT_OUTPUT   = "outputs/forecast_plot.png"
FORECAST_CSV_OUTPUT    = "outputs/forecast_24h.csv"
FORECAST_ALERTS_OUTPUT = "outputs/forecast_alerts.csv"

# Model parameters - PART 3: Prophet Forecasting
UNHEALTHY_THRESHOLD = 100  # PM2.5 ≥ 100 µg/m³ = "Unhealthy" — minimum alert trigger
HAZARDOUS_THRESHOLD = 150  # PM2.5 ≥ 150 µg/m³ = "Hazardous" (US EPA AQI scale)
FORECAST_HOURS = 24        # How many hours ahead to predict

# ============================================================================
# FUNCTION 1: Load and Prepare Data
# ============================================================================

def load_and_prepare_data(file_path):
    """
    Load the cleaned data and prepare it for machine learning

    This function:
    1. Reads the CSV file
    2. Removes any rows with missing values
    3. Filters out invalid PM2.5 readings (negative values)

    Parameters:
        file_path (str): Path to the cleaned data CSV file

    Returns:
        pandas.DataFrame: The loaded and cleaned dataframe
    """
    print("=" * 70)
    print("STEP 1: Loading and Preparing Data")
    print("=" * 70)

    # Check if file exists
    if not os.path.exists(file_path):
        print(f"✗ Error: File not found at {file_path}")
        print("Please run src/preprocess.py first")
        return None

    # Load the CSV file
    df = pd.read_csv(file_path)
    print(f"✓ Loaded {len(df)} rows from {file_path}")

    # Show initial data info
    print(f"\nInitial dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")

    # Remove rows with missing values
    # Missing values can cause errors in machine learning models
    initial_rows = len(df)
    df = df.dropna()
    print(f"\n✓ Removed {initial_rows - len(df)} rows with missing values")

    # Filter out invalid PM2.5 readings
    # Negative values are sensor errors or data quality issues
    df = df[df['pm25'] > 0]
    print(f"✓ Filtered out invalid PM2.5 readings (negative values)")
    print(f"Final dataset shape: {df.shape}")

    return df

# ============================================================================
# FUNCTION 2: Define Features and Target
# ============================================================================

def define_features_and_target(df):
    """
    Select which columns to use as features (inputs) and target (output)

    FEATURES (X): The input variables the model uses to make predictions
    - pm25: The main air quality measurement
    - hour: Time of day (traffic patterns vary by hour)
    - day_of_week: Day patterns (weekdays vs weekends)
    - month: Seasonal patterns (winter smog vs summer)
    - is_weekend: Binary indicator for weekend days
    - pm25_24h_avg: Smoothed trend over 24 hours

    TARGET (y): What we're trying to predict
    - aqi_category: The danger level (Good, Moderate, Unhealthy, Hazardous)

    Parameters:
        df (pandas.DataFrame): The prepared dataframe

    Returns:
        tuple: (X, y) where X is features and y is target
    """
    print("\n" + "=" * 70)
    print("STEP 2: Defining Features and Target")
    print("=" * 70)

    # Define feature columns (inputs to the model)
    feature_columns = [
        'pm25',           # Primary pollutant measurement
        'hour',           # Time of day
        'day_of_week',    # Day of the week
        'month',          # Month of the year
        'is_weekend',     # Weekend flag
        'pm25_24h_avg'    # Rolling average
    ]

    # Check if optional features exist (temperature, humidity)
    # These might not be in our dataset, so we check first
    if 'temperature' in df.columns:
        feature_columns.append('temperature')
        print("✓ Found 'temperature' column, adding to features")

    if 'humidity' in df.columns:
        feature_columns.append('humidity')
        print("✓ Found 'humidity' column, adding to features")

    # Extract features (X) and target (y)
    X = df[feature_columns]  # Features: what the model learns from
    y = df['aqi_category']   # Target: what the model predicts

    print(f"\n✓ Selected {len(feature_columns)} features:")
    for i, col in enumerate(feature_columns, 1):
        print(f"  {i}. {col}")

    print(f"\n✓ Target variable: aqi_category")
    print(f"  Classes: {y.unique()}")
    print(f"  Distribution:")
    for category, count in y.value_counts().items():
        percentage = (count / len(y)) * 100
        print(f"    {category}: {count} ({percentage:.2f}%)")

    return X, y, feature_columns

# ============================================================================
# FUNCTION 3: Split Data into Training and Testing Sets
# ============================================================================

def split_data(X, y, test_size=0.2, random_state=42):
    """
    Split the data into training and testing sets

    WHY DO WE SPLIT THE DATA?
    - Training set: Used to teach the model patterns
    - Testing set: Used to evaluate how well the model learned
    - We NEVER let the model see the test data during training
    - This simulates how the model will perform on new, unseen data

    WHAT IS random_state?
    - It's a seed for the random number generator
    - Using the same random_state (e.g., 42) ensures we get the same split
      every time we run the code
    - This makes our results reproducible - important for science!
    - The number 42 is arbitrary (it's a joke from "Hitchhiker's Guide to the Galaxy")
    - You could use any number: 42, 123, 999, etc.

    Parameters:
        X (DataFrame): Features
        y (Series): Target variable
        test_size (float): Proportion of data for testing (0.2 = 20%)
        random_state (int): Random seed for reproducibility

    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    print("\n" + "=" * 70)
    print("STEP 3: Splitting Data into Training and Testing Sets")
    print("=" * 70)

    # Split the data
    # test_size=0.2 means 20% for testing, 80% for training
    # stratify=y ensures each class is proportionally represented in both sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y  # Maintains class distribution in both sets
    )

    print(f"✓ Data split with test_size={test_size} (random_state={random_state})")
    print(f"\nTraining set:")
    print(f"  Samples: {len(X_train)} ({(1-test_size)*100:.0f}%)")
    print(f"  Features: {X_train.shape[1]}")

    print(f"\nTesting set:")
    print(f"  Samples: {len(X_test)} ({test_size*100:.0f}%)")
    print(f"  Features: {X_test.shape[1]}")

    print(f"\nClass distribution in training set:")
    for category, count in y_train.value_counts().items():
        percentage = (count / len(y_train)) * 100
        print(f"  {category}: {count} ({percentage:.2f}%)")

    return X_train, X_test, y_train, y_test

# ============================================================================
# FUNCTION 4: Train Random Forest Model
# ============================================================================

def train_random_forest(X_train, y_train, n_estimators=100, random_state=42):
    """
    Train a Random Forest Classifier

    WHAT IS n_estimators?
    - It's the number of decision trees in the forest
    - More trees = more accurate but slower to train
    - 100 is a good default balance between accuracy and speed
    - Common values: 50, 100, 200, 500

    HOW DOES TRAINING WORK?
    1. The algorithm creates n_estimators trees (e.g., 100 trees)
    2. Each tree is trained on a random subset of the data
    3. Each tree learns different patterns
    4. When predicting, all trees vote and majority wins

    Parameters:
        X_train (DataFrame): Training features
        y_train (Series): Training target
        n_estimators (int): Number of trees in the forest
        random_state (int): Random seed for reproducibility

    Returns:
        RandomForestClassifier: The trained model
    """
    print("\n" + "=" * 70)
    print("STEP 4: Training Random Forest Classifier")
    print("=" * 70)

    print(f"\nModel configuration:")
    print(f"  Algorithm: Random Forest Classifier")
    print(f"  Number of trees (n_estimators): {n_estimators}")
    print(f"  Random state: {random_state}")

    # Create the Random Forest model
    # n_estimators: number of trees
    # random_state: for reproducibility
    # n_jobs=-1: use all CPU cores for faster training
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1,  # Use all available CPU cores
        verbose=1   # Show progress during training
    )

    print(f"\nTraining model on {len(X_train)} samples...")
    print("This may take a minute...")

    # Train the model
    # .fit() is where the actual learning happens
    model.fit(X_train, y_train)

    print("\n✓ Model training complete!")

    return model

# ============================================================================
# FUNCTION 5: Evaluate Model Performance
# ============================================================================

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the trained model on the test set

    METRICS EXPLAINED:
    - Accuracy: Overall correctness (correct predictions / total predictions)
    - Precision: Of all predicted "Hazardous", how many were actually hazardous?
    - Recall: Of all actual "Hazardous" cases, how many did we catch?
    - F1-Score: Harmonic mean of precision and recall (balanced metric)

    Parameters:
        model: The trained model
        X_test (DataFrame): Test features
        y_test (Series): Test target

    Returns:
        tuple: (accuracy, y_pred) - accuracy score and predictions
    """
    print("\n" + "=" * 70)
    print("STEP 5: Evaluating Model Performance")
    print("=" * 70)

    # Make predictions on the test set
    # The model has never seen this data before
    print(f"\nMaking predictions on {len(X_test)} test samples...")
    y_pred = model.predict(X_test)

    # Calculate accuracy
    # Accuracy = (correct predictions) / (total predictions)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"\n{'='*50}")
    print(f"OVERALL ACCURACY: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"{'='*50}")

    # Print detailed classification report
    # This shows precision, recall, and F1-score for each class
    print("\nDETAILED CLASSIFICATION REPORT:")
    print("-" * 50)
    print(classification_report(y_test, y_pred))

    print("\nMETRICS EXPLANATION:")
    print("  Precision: Of all predicted X, how many were actually X?")
    print("  Recall: Of all actual X cases, how many did we predict?")
    print("  F1-Score: Balanced measure of precision and recall")
    print("  Support: Number of actual occurrences of each class")

    return accuracy, y_pred

# ============================================================================
# FUNCTION 6: Plot and Save Confusion Matrix
# ============================================================================

def plot_confusion_matrix(y_test, y_pred, output_path):
    """
    Create and save a confusion matrix visualization

    WHAT IS A CONFUSION MATRIX?
    A table showing where the model gets "confused":
    - Rows: Actual true labels
    - Columns: Predicted labels
    - Diagonal: Correct predictions
    - Off-diagonal: Mistakes

    Example:
                  Predicted
                  Good  Hazardous
    Actual Good    90      10       (90 correct, 10 wrong)
           Haz     5       95       (5 wrong, 95 correct)

    Parameters:
        y_test (Series): True labels
        y_pred (array): Predicted labels
        output_path (str): Where to save the plot
    """
    print("\n" + "=" * 70)
    print("STEP 6: Creating Confusion Matrix")
    print("=" * 70)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Calculate confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # Get unique class labels in sorted order
    labels = sorted(y_test.unique())

    # Create a figure
    plt.figure(figsize=(10, 8))

    # Create heatmap
    # annot=True: show numbers in cells
    # fmt='d': format numbers as integers
    # cmap: color scheme (Blues = light to dark blue)
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=labels,
        yticklabels=labels,
        cbar_kws={'label': 'Count'}
    )

    plt.title('Confusion Matrix - AQI Category Prediction', fontsize=16, fontweight='bold')
    plt.ylabel('Actual Category', fontsize=12)
    plt.xlabel('Predicted Category', fontsize=12)
    plt.tight_layout()

    # Save the plot
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Confusion matrix saved to: {output_path}")

    plt.close()

    # Print interpretation
    print("\nCONFUSION MATRIX INTERPRETATION:")
    print("  Diagonal values (top-left to bottom-right) = Correct predictions")
    print("  Off-diagonal values = Misclassifications")
    print("  Higher diagonal values = Better model performance")

# ============================================================================
# FUNCTION 7: Plot and Save Feature Importance
# ============================================================================

def plot_feature_importance(model, feature_names, output_path):
    """
    Create and save a feature importance chart

    WHAT IS FEATURE IMPORTANCE?
    - Shows which features the model relies on most for predictions
    - Higher importance = more influential in decision-making
    - Helps us understand what drives air quality predictions

    For example:
    - pm25 might have 60% importance (most important)
    - hour might have 15% importance
    - is_weekend might have 5% importance (least important)

    Parameters:
        model: The trained Random Forest model
        feature_names (list): Names of the features
        output_path (str): Where to save the plot
    """
    print("\n" + "=" * 70)
    print("STEP 7: Creating Feature Importance Chart")
    print("=" * 70)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Get feature importances from the model
    # Random Forest calculates this automatically during training
    importances = model.feature_importances_

    # Create a dataframe for easier plotting
    feature_importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)

    print("\nFEATURE IMPORTANCE RANKING:")
    print("-" * 50)
    for idx, row in feature_importance_df.iterrows():
        print(f"  {row['feature']:20} : {row['importance']:.4f} ({row['importance']*100:.2f}%)")

    # Create the plot
    plt.figure(figsize=(10, 6))

    # Create horizontal bar chart
    plt.barh(
        feature_importance_df['feature'],
        feature_importance_df['importance'],
        color='steelblue'
    )

    plt.xlabel('Importance Score', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    plt.title('Feature Importance in AQI Prediction', fontsize=16, fontweight='bold')
    plt.tight_layout()

    # Save the plot
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Feature importance chart saved to: {output_path}")

    plt.close()

    print("\nINTERPRETATION:")
    print("  Features at the top are most important for predictions")
    print("  Features at the bottom have less influence on the model")

# ============================================================================
# FUNCTION 8: Save the Trained Model
# ============================================================================

def save_model(model, output_path):
    """
    Save the trained model to disk using joblib

    WHAT IS A .pkl FILE?
    - .pkl stands for "pickle" - Python's way of saving objects
    - It serializes (converts) the model into a file format
    - Later, we can load this file and use the model without retraining
    - Think of it like saving a video game - you can resume later

    WHY USE JOBLIB INSTEAD OF PICKLE?
    - joblib is optimized for large numpy arrays (which ML models use)
    - It's faster and more efficient for machine learning models
    - It's the standard in the scikit-learn ecosystem

    Parameters:
        model: The trained model to save
        output_path (str): Where to save the model file
    """
    print("\n" + "=" * 70)
    print("STEP 8: Saving Trained Model")
    print("=" * 70)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save the model using joblib
    # compress=3: moderate compression (balance between size and speed)
    joblib.dump(model, output_path, compress=3)

    # Get file size
    file_size = os.path.getsize(output_path) / 1024  # Convert to KB

    print(f"✓ Model saved to: {output_path}")
    print(f"  File size: {file_size:.2f} KB")
    print(f"  Format: Joblib pickle (.pkl)")

    print("\nTO LOAD THIS MODEL LATER:")
    print(f"  model = joblib.load('{output_path}')")
    print(f"  predictions = model.predict(new_data)")

# ============================================================================
# ============================================================================
# PART 2: ISOLATION FOREST - ANOMALY DETECTION
# ============================================================================
# ============================================================================

"""
WHY USE ISOLATION FOREST INSTEAD OF SIMPLE THRESHOLD RULES?
============================================================

Simple Threshold Approach:
- "If PM2.5 > 200, it's an anomaly"
- Problem: What if PM2.5 is normally 180-220 during rush hour?
- Problem: What if PM2.5 suddenly jumps from 50 to 150? (3x increase!)
- A fixed threshold misses context and patterns

Isolation Forest Approach:
- Learns what "normal" looks like from the data
- Considers multiple features together (PM2.5, time, rolling average)
- Detects unusual PATTERNS, not just high values
- Examples it can catch:
  * PM2.5 = 180 at 3am (unusual for that time)
  * PM2.5 spikes 100 points in 1 hour (unusual rate of change)
  * PM2.5 = 50 when rolling average is 200 (unusual drop)

HOW DOES ISOLATION FOREST WORK?
================================
Think of it like finding the "odd one out" in a crowd:
- Normal data points are surrounded by similar neighbors
- Anomalies are isolated - they're far from the crowd
- The algorithm builds trees that try to isolate each point
- Points that are easy to isolate (few splits needed) = anomalies
- Points that are hard to isolate (many splits needed) = normal

Example:
- Normal: PM2.5=150, hour=8am, rolling_avg=145 (rush hour pattern)
- Anomaly: PM2.5=150, hour=3am, rolling_avg=50 (weird spike at night)
"""

# ============================================================================
# FUNCTION 9: Load Data for Anomaly Detection
# ============================================================================

def load_data_for_anomaly_detection(file_path):
    """
    Load the cleaned data CSV for the anomaly detection pipeline.

    We need 'split', 'city_season', and all 8 pollutant columns to be present
    because the per-group Isolation Forest trained only on training rows and
    then applies predictions to both train + test rows.

    Parameters:
        file_path (str): Path to cleaned_data.csv

    Returns:
        pandas.DataFrame: Loaded data, or None if file is missing
    """
    print("\n" + "=" * 70)
    print("PART 2: CITY+SEASON-AWARE ANOMALY DETECTION")
    print("=" * 70)
    print("\nSTEP 1: Loading Data for Anomaly Detection")
    print("-" * 50)

    if not os.path.exists(file_path):
        print(f"✗ File not found: {file_path}")
        print("  Run src/preprocess.py first.")
        return None

    df = pd.read_csv(file_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    print(f"✓ Loaded {len(df):,} rows")
    print(f"  Cities    : {sorted(df['city'].unique())}")
    print(f"  Seasons   : {sorted(df['season'].unique())}")
    print(f"  Splits    : {df['split'].value_counts().to_dict()}")
    print(f"  Groups    : {df['city_season'].nunique()} city-season groups")
    return df


# ============================================================================
# FUNCTION 10: Train Per-Group Isolation Forests
# ============================================================================

def train_per_group_isolation_forests(df, contamination=0.05):
    """
    Train one Isolation Forest per city_season group using ONLY training data.

    WHY ONE MODEL PER CITY-SEASON GROUP?
    ======================================
    A single global Isolation Forest has a critical blind spot:
    - Lahore in Winter: PM2.5 = 300 µg/m³ is "normal" (dense winter smog)
    - Karachi in Summer: PM2.5 = 300 µg/m³ is a genuine emergency spike
    A global model trained on both would call Lahore-Winter spikes "normal"
    because they match Lahore's typical winter readings — missing real events.

    By training a separate model for each (city, season) pair, each model
    learns what "normal" means specifically in that context.

    Training uses only the 'train' split rows (before 2024-07-01) so the
    model never sees future data during training — correct ML practice.

    Features used per group: pm25, pm10, pm25_24h_avg, hour

    Parameters:
        df (pandas.DataFrame): Cleaned data with 'city_season' and 'split' columns
        contamination (float): Expected fraction of anomalies per group (0.05 = 5%)

    Returns:
        pandas.DataFrame: Original DataFrame with a new 'is_anomaly' column added
    """
    print("\n" + "=" * 70)
    print("STEP 2: Training Per-Group Isolation Forests")
    print("-" * 50)
    print(f"  Features  : pm25, pm10, pm25_24h_avg, hour")
    print(f"  Contamination : {contamination} ({contamination*100:.0f}% expected anomalies)")
    print(f"  Train-only fit: models learn from pre-2024-07-01 rows only")

    # Features each group model learns from
    # pm25 + pm10: raw pollutant levels (the spike signal)
    # pm25_24h_avg: recent trend (spike relative to background)
    # hour: time of day context (3am spike is more suspicious than 8am rush)
    anomaly_features = ['pm25', 'pm10', 'pm25_24h_avg', 'hour']

    # We'll store each row's anomaly label here; default is 0 (normal)
    df = df.copy()
    df['is_anomaly'] = 0

    groups_trained = 0
    groups_skipped = 0
    total_anomalies = 0

    # Loop over every city-season combination (e.g. "Lahore_Winter", "Karachi_Summer")
    for group_name, group_data in df.groupby('city_season'):

        # Split this group into train and all rows
        train_rows = group_data[group_data['split'] == 'train']
        all_rows   = group_data  # We'll predict on both train + test rows

        # Skip groups that don't have enough training rows to fit a model
        # A model trained on fewer than 100 rows is unreliable
        if len(train_rows) < MIN_ROWS_PER_GROUP:
            print(f"  ✗ Skipping {group_name:30} — only {len(train_rows)} train rows "
                  f"(need {MIN_ROWS_PER_GROUP})")
            groups_skipped += 1
            continue

        # Extract feature matrices for fitting and predicting
        X_train = train_rows[anomaly_features].fillna(0)
        X_all   = all_rows[anomaly_features].fillna(0)

        # Create and fit the Isolation Forest for this group only
        group_model = IsolationForest(
            contamination=contamination,
            random_state=RANDOM_STATE,
            n_estimators=100,
            n_jobs=-1
        )
        group_model.fit(X_train)   # Train on train rows only

        # Predict on ALL rows in this group (returns -1=anomaly, 1=normal)
        preds = group_model.predict(X_all)

        # Write anomaly labels back to the main DataFrame using original row indices
        # (predict == -1) is True for anomalies → cast to int → 1=anomaly, 0=normal
        df.loc[all_rows.index, 'is_anomaly'] = (preds == -1).astype(int)

        # Save this group's model to disk for later use by the alert system
        model_filename = f"isolation_forest_{group_name}.pkl"
        model_path = os.path.join(ISOLATION_FOREST_DIR, model_filename)
        os.makedirs(ISOLATION_FOREST_DIR, exist_ok=True)
        joblib.dump(group_model, model_path, compress=3)

        n_group_anomalies = (preds == -1).sum()
        total_anomalies  += n_group_anomalies
        groups_trained   += 1

        print(f"  ✓ {group_name:30}  train={len(train_rows):>5}  "
              f"anomalies={n_group_anomalies:>4}  → {model_filename}")

    print(f"\n✓ Done: {groups_trained} groups trained, {groups_skipped} skipped")
    print(f"  Total anomalies across all groups: {total_anomalies:,}")
    print(f"  Models saved to: {ISOLATION_FOREST_DIR}/isolation_forest_*.pkl")
    return df


# ============================================================================
# FUNCTION 11: Plot Anomalies — Multi-City Subplots
# ============================================================================

def plot_anomalies(df, output_path):
    """
    Create a multi-city subplot figure showing PM2.5 readings over time with
    anomalies highlighted as red scatter points.

    Each city gets its own subplot row so patterns remain readable
    (mixing all cities into one chart makes it unreadable).

    A vertical dashed line marks the train/test boundary (2024-07-01),
    making it easy to see whether anomalies cluster in the test window.

    Parameters:
        df (pandas.DataFrame): Full dataset with 'is_anomaly' column added
        output_path (str): File path where the PNG should be saved
    """
    print("\n" + "=" * 70)
    print("STEP 3: Plotting Anomaly Timeline (Multi-City)")
    print("-" * 50)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    cities = sorted(df['city'].unique())
    n_cities = len(cities)

    # One subplot row per city; shared x-axis so dates align across subplots
    fig, axes = plt.subplots(
        n_cities, 1,
        figsize=(18, 4 * n_cities),
        sharex=True
    )
    if n_cities == 1:
        axes = [axes]  # Make it iterable when there's only one city

    # The vertical line marking where training data ends
    split_boundary = pd.Timestamp("2024-07-01")

    for ax, city in zip(axes, cities):
        city_data = df[df['city'] == city].sort_values('timestamp')

        normal_data  = city_data[city_data['is_anomaly'] == 0]
        anomaly_data = city_data[city_data['is_anomaly'] == 1]

        # Blue line: normal hourly PM2.5 readings
        ax.plot(
            normal_data['timestamp'],
            normal_data['pm25'],
            color='steelblue',
            linewidth=0.6,
            alpha=0.7,
            label='Normal'
        )

        # Red dots: anomalous readings on top of the line
        ax.scatter(
            anomaly_data['timestamp'],
            anomaly_data['pm25'],
            color='red',
            s=12,
            alpha=0.7,
            zorder=5,
            label=f'Anomaly (n={len(anomaly_data):,})'
        )

        # Dashed vertical line: train/test split boundary
        ax.axvline(
            x=split_boundary,
            color='gray',
            linewidth=1.5,
            linestyle='--',
            alpha=0.8,
            label='Train/Test split'
        )

        ax.set_ylabel('PM2.5 (µg/m³)', fontsize=9)
        ax.set_title(f'{city}', fontsize=11, fontweight='bold')
        ax.legend(loc='upper left', fontsize=8, framealpha=0.7)
        ax.grid(True, alpha=0.25)

    axes[-1].set_xlabel('Date', fontsize=11)
    fig.suptitle(
        'PM2.5 Anomaly Detection — Per City (red dots = detected anomalies)',
        fontsize=14, fontweight='bold', y=1.01
    )
    plt.tight_layout()

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Anomaly plot saved: {output_path}")
    plt.close()

    # Print a quick per-city anomaly count summary
    print("\nAnomalies per city:")
    for city in cities:
        city_data  = df[df['city'] == city]
        n_anomaly  = city_data['is_anomaly'].sum()
        n_test_anomaly = city_data[
            (city_data['is_anomaly'] == 1) & (city_data['split'] == 'test')
        ].shape[0]
        print(f"  {city:12}: {n_anomaly:>5,} total  ({n_test_anomaly:>4,} in test window)")


# ============================================================================
# FUNCTION 12: Save Anomalies to CSV
# ============================================================================

def save_anomalies(df, output_path):
    """
    Save rows flagged as anomalies to a CSV with all 8 pollutant columns.

    Keeping all 8 pollutant columns is required because Phase 3
    (source_classifier.py) uses the chemical signature of each anomaly
    (ratios and z-scores across pollutants) to determine the emission source.

    Parameters:
        df (pandas.DataFrame): Full dataset with 'is_anomaly' column
        output_path (str): Path to save the anomalies CSV
    """
    print("\n" + "=" * 70)
    print("STEP 4: Saving Anomalies CSV")
    print("-" * 50)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    anomalies = df[df['is_anomaly'] == 1].copy()
    anomalies = anomalies.sort_values(['city', 'timestamp'])

    anomalies.to_csv(output_path, index=False)

    print(f"✓ Saved {len(anomalies):,} anomalous rows to: {output_path}")
    print(f"  Columns: {list(anomalies.columns)}")

    # Verify all 8 pollutant columns are present (required for Phase 3)
    required_pollutants = ['pm25', 'pm10', 'no', 'no2', 'so2', 'nh3', 'co', 'o3']
    present = [c for c in required_pollutants if c in anomalies.columns]
    missing = [c for c in required_pollutants if c not in anomalies.columns]
    print(f"\n  Pollutant columns present : {present}")
    if missing:
        print(f"  Pollutant columns MISSING : {missing}  ← fix before running source_classifier.py")

    # Show anomaly breakdown by city and split
    print("\nAnomaly breakdown (city × split):")
    breakdown = anomalies.groupby(['city', 'split']).size().reset_index(name='count')
    print(breakdown.to_string(index=False))

    # Phase 2 checkpoint: test-window anomalies must exist
    test_anomalies = anomalies[anomalies['split'] == 'test']
    print(f"\nTest-window anomalies (Jul–Dec 2024): {len(test_anomalies):,}")
    if len(test_anomalies) == 0:
        print("  ⚠ Warning: no test anomalies found — check split column and date parsing")


# ============================================================================
# FUNCTION 13: Main Function for Anomaly Detection
# ============================================================================

def main_anomaly_detection():
    """
    Orchestrate the full per-group anomaly detection pipeline (Phase 2).
    """
    print("\n" + "=" * 70)
    print("PART 2: CITY+SEASON-AWARE ANOMALY DETECTION")
    print("=" * 70)

    # Step 1: Load cleaned data
    df = load_data_for_anomaly_detection(INPUT_FILE)
    if df is None:
        return

    # Step 2: Train one Isolation Forest per city_season group (train split only)
    df_with_anomalies = train_per_group_isolation_forests(df, contamination=CONTAMINATION)

    # Step 3: Multi-city subplot of anomalies over time
    plot_anomalies(df_with_anomalies, ANOMALY_PLOT_OUTPUT)

    # Step 4: Save anomalies CSV with all 8 pollutant columns
    save_anomalies(df_with_anomalies, ANOMALIES_OUTPUT)

    # Final summary
    print("\n" + "=" * 70)
    print("✓ ANOMALY DETECTION COMPLETE!")
    print("=" * 70)
    print(f"\nGenerated files:")
    print(f"  Models   : models/isolation_forest_{{city}}_{{season}}.pkl")
    print(f"  Plot     : {ANOMALY_PLOT_OUTPUT}")
    print(f"  Anomalies: {ANOMALIES_OUTPUT}")
    print(f"\nNext step: python src/source_classifier.py")

# ============================================================================
# ============================================================================
# PART 3: PROPHET — TIME-SERIES FORECASTING
# ============================================================================
# ============================================================================

"""
WHAT IS FACEBOOK PROPHET?
=========================
Prophet is a forecasting library created by Facebook (Meta) in 2017.
It's designed for messy, real-world time-series data — exactly what air
quality sensor readings look like.

Why use Prophet instead of a simple trend line?
- Automatically learns daily, weekly, and yearly seasonal patterns
  (e.g. PM2.5 spikes every weekday morning during rush hour)
- Handles missing timestamps gracefully (common in sensor networks)
- Returns a built-in confidence interval showing forecast uncertainty
- Requires almost no hyperparameter tuning compared to other models

In our case, Prophet learns patterns like:
- "PM2.5 is always higher at 8am and 6pm (rush hour peaks)"
- "PM2.5 is higher in November–January (winter temperature inversions)"
"""

# ============================================================================
# FUNCTION 14 (helper): Classify a PM2.5 value into an alert colour level
# ============================================================================

def assign_forecast_alert_level(pm25_value):
    """
    Map a PM2.5 concentration to one of four SmogAlert colour levels.

    This is the same scale used across the pipeline so every module is
    consistent.  Thresholds are based on the US EPA AQI breakpoints.

    Parameters:
        pm25_value (float): PM2.5 concentration in µg/m³

    Returns:
        str: 'GREEN', 'YELLOW', 'ORANGE', or 'RED'
    """
    if pm25_value <= 50:
        return 'GREEN'       # Good — air quality is satisfactory
    elif pm25_value <= 100:
        return 'YELLOW'      # Moderate — acceptable for most people
    elif pm25_value <= 150:
        return 'ORANGE'      # Unhealthy for sensitive groups
    else:
        return 'RED'         # Hazardous — serious health risk for all


# ============================================================================
# FUNCTION 15: Train Prophet and forecast for ONE city
# ============================================================================

def forecast_city_with_prophet(city_df, city_name, forecast_hours=24):
    """
    Train a Prophet model on one city's PM2.5 history and produce a
    forecast for the next `forecast_hours` hours.

    WHY ONE MODEL PER CITY?
    =======================
    A model trained on all five cities lumps together very different patterns:
    - Lahore in winter regularly exceeds 300 µg/m³ (dense seasonal smog)
    - Karachi in summer sits around 50–80 µg/m³ (sea-breeze effect)
    Mixing these flattens the seasonal signal for each city.  Training a
    separate Prophet per city lets each model learn the exact daily, weekly,
    and yearly cycles specific to that location's climate and traffic.

    Parameters:
        city_df    (pd.DataFrame): Rows for this city from cleaned_data.csv.
                                   Must have 'timestamp' and 'pm25' columns.
        city_name  (str):          City label (e.g. 'Lahore') used for saving
                                   the model and labelling outputs.
        forecast_hours (int):      How many hours ahead to forecast (default 24).

    Returns:
        tuple: (future_forecast_df, last_training_ts) where
               future_forecast_df has columns ds, yhat, yhat_lower, yhat_upper
               and last_training_ts is the final historical timestamp.
    """

    # ---- Prepare Prophet input ----
    # Prophet requires exactly two columns named 'ds' (datestamp) and 'y'
    # (the value to forecast).  Any other column names raise a ValueError.
    prophet_df = city_df[['timestamp', 'pm25']].copy()
    prophet_df = prophet_df.rename(columns={'timestamp': 'ds', 'pm25': 'y'})
    prophet_df = prophet_df.sort_values('ds').reset_index(drop=True)

    # Remove negative/zero PM2.5 readings (sensor errors)
    prophet_df = prophet_df[prophet_df['y'] > 0]

    print(f"  {city_name:12}: {len(prophet_df):>6,} rows  "
          f"({prophet_df['ds'].min().date()} → {prophet_df['ds'].max().date()})")

    # ---- Train Prophet ----
    # interval_width=0.95 → 95% confidence band in the forecast
    model = Prophet(
        daily_seasonality=True,   # Rush-hour peaks within each day
        weekly_seasonality=True,  # Weekday vs. weekend traffic differences
        yearly_seasonality=True,  # Winter smog season vs. summer
        interval_width=0.95,      # 95% confidence interval
    )
    model.fit(prophet_df)

    # ---- Save this city's model ----
    model_path = f"models/prophet_{city_name}.pkl"
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, model_path, compress=3)

    # ---- Build the future dataframe and predict ----
    # make_future_dataframe returns historical timestamps PLUS the next
    # forecast_hours hourly slots beyond the last observed data point.
    future_df = model.make_future_dataframe(periods=forecast_hours, freq='h')
    forecast   = model.predict(future_df)

    last_training_ts = prophet_df['ds'].max()

    # Keep only the genuinely future rows (timestamps after last observation)
    future_forecast = forecast[forecast['ds'] > last_training_ts].copy()

    return future_forecast, last_training_ts


# ============================================================================
# FUNCTION 16: Build forecast-based alert records for one city
# ============================================================================

def build_forecast_alerts(future_forecast, city_name, reference_ts):
    """
    Scan a 24-hour forecast and emit one alert record per contiguous block
    of hours where the predicted PM2.5 exceeds the Unhealthy threshold.

    WHY ONE ALERT PER BLOCK (NOT PER HOUR)?
    ========================================
    If the forecast shows 6 consecutive Unhealthy hours, emitting 6 identical
    alerts creates noise and alert fatigue.  Instead we find the first hour of
    each above-threshold block and emit a single forward-looking alert that
    tells residents: "it's going to get bad at [time], roughly N hours from now."
    This is how operational weather warning systems work.

    Alert trigger levels:
      PM2.5 ≥ 100 µg/m³ (Unhealthy)  → ORANGE forecast alert
      PM2.5 ≥ 150 µg/m³ (Hazardous)  → RED    forecast alert

    Parameters:
        future_forecast (pd.DataFrame): Prophet prediction for future hours;
                                        must have columns 'ds', 'yhat', 'yhat_upper'.
        city_name       (str):          City name injected into alert text.
        reference_ts    (pd.Timestamp): The last historical timestamp (= "now"
                                        from the model's perspective), used to
                                        calculate hours-ahead values.

    Returns:
        list[dict]: Alert records ready to be converted to a DataFrame.
                    Returns an empty list if no threshold is breached.
    """

    # ---- Bilingual alert templates (forward-looking language) ----
    # ORANGE: predicted PM2.5 ≥ 100 and < 150 (Unhealthy for sensitive groups)
    ORANGE_EN = (
        "⚠ Forecast Alert — {city}. "
        "PM2.5 is predicted to reach {pm25:.0f} µg/m³ at {time} "
        "(approximately {hours} hour(s) from now), entering Unhealthy levels. "
        "Children, elderly residents, and people with respiratory conditions "
        "should prepare to limit outdoor activity and keep windows closed."
    )
    ORANGE_UR = (
        "⚠ پیشگوئی الرٹ — {city}۔ "
        "PM2.5 کی سطح {time} پر {pm25:.0f} µg/m³ تک پہنچنے کی توقع ہے "
        "(ابھی سے تقریباً {hours} گھنٹے بعد)، جو غیر صحت بخش حد میں ہوگی۔ "
        "بچے، بزرگ اور سانس کے مریض باہر جانے کی سرگرمیاں کم کرنے کی تیاری کریں "
        "اور کھڑکیاں بند رکھیں۔"
    )

    # RED: predicted PM2.5 ≥ 150 (Hazardous — serious risk for all residents)
    RED_EN = (
        "🚨 Forecast Alert — {city}. "
        "PM2.5 is forecast to reach {pm25:.0f} µg/m³ at {time} "
        "(approximately {hours} hour(s) from now), entering Hazardous territory. "
        "All residents should arrange to stay indoors, avoid all outdoor activity, "
        "and use an air purifier or N95 mask if going outside is unavoidable."
    )
    RED_UR = (
        "🚨 پیشگوئی الرٹ — {city}۔ "
        "PM2.5 {time} پر {pm25:.0f} µg/m³ تک پہنچنے کی پیشگوئی ہے "
        "(ابھی سے تقریباً {hours} گھنٹے بعد)، جو خطرناک سطح ہے۔ "
        "تمام شہری گھر کے اندر رہنے کا انتظام کریں، ہر قسم کی بیرونی سرگرمی سے گریز کریں "
        "اور باہر جانا ناگزیر ہو تو N95 ماسک یا ایئر پیوریفائر استعمال کریں۔"
    )

    alert_records = []
    # Use yhat (point estimate) to decide whether a breach occurs.
    # yhat_upper (worst-case bound) is stored in the record so the dashboard
    # can show the conservative scenario alongside the point estimate.
    above_threshold = future_forecast['yhat'] >= UNHEALTHY_THRESHOLD

    in_block   = False  # Are we currently inside an above-threshold block?
    block_level = None  # 'ORANGE' or 'RED' for the current block

    for _, row in future_forecast.iterrows():
        predicted  = row['yhat']
        upper      = row['yhat_upper']
        ts         = row['ds']
        hours_ahead = int(round((ts - reference_ts).total_seconds() / 3600))
        time_label  = ts.strftime('%Y-%m-%d %H:%M')

        if predicted >= UNHEALTHY_THRESHOLD:
            level = 'RED' if predicted >= HAZARDOUS_THRESHOLD else 'ORANGE'

            if not in_block:
                # ---- First hour of a new breach block → emit alert ----
                in_block    = True
                block_level = level

                if level == 'RED':
                    en = RED_EN.format(city=city_name, pm25=predicted,
                                       time=time_label, hours=hours_ahead)
                    ur = RED_UR.format(city=city_name, pm25=predicted,
                                       time=time_label, hours=hours_ahead)
                    aff = "All residents"
                    act = "Stay indoors; avoid all outdoor activity; use N95 mask or air purifier"
                else:
                    en = ORANGE_EN.format(city=city_name, pm25=predicted,
                                          time=time_label, hours=hours_ahead)
                    ur = ORANGE_UR.format(city=city_name, pm25=predicted,
                                          time=time_label, hours=hours_ahead)
                    aff = "Children, elderly, respiratory patients"
                    act = "Limit outdoor activity; keep windows closed"

                alert_records.append({
                    'alert_type':       'forecast',
                    'city':             city_name,
                    'forecast_timestamp': time_label,
                    'hours_ahead':      hours_ahead,
                    'pm25_predicted':   round(predicted, 1),
                    'pm25_upper':       round(upper, 1),
                    'aqi_level':        'Hazardous' if level == 'RED' else 'Unhealthy',
                    'alert_level':      level,
                    'affected_groups':  aff,
                    'protective_actions': act,
                    'alert_text_en':    en,
                    'alert_text_ur':    ur,
                })
            # If still in a block but level has escalated to RED, do nothing —
            # the opening alert for this block already captured the breach.
        else:
            # Below threshold: reset block tracking
            in_block    = False
            block_level = None

    return alert_records


# ============================================================================
# FUNCTION 17: Forecast all five cities and generate forecast alerts
# ============================================================================

def forecast_all_cities_with_prophet(file_path, forecast_hours=24):
    """
    Run the full city-specific Prophet pipeline:
      1. Load cleaned_data.csv
      2. For each city: train a dedicated Prophet model, forecast next 24 hours
      3. Combine all city forecasts into one CSV
      4. Generate forward-looking alerts for any city breaching PM2.5 thresholds
      5. Save a multi-city subplot chart

    Output files:
      outputs/forecast_24h.csv     — combined forecast with 'city' column
      outputs/forecast_alerts.csv  — forecast-based bilingual alerts
      outputs/forecast_plot.png    — 5-panel city subplot
      models/prophet_{city}.pkl    — one saved model per city

    Parameters:
        file_path      (str): Path to cleaned_data.csv
        forecast_hours (int): Hours ahead to forecast per city (default 24)

    Returns:
        pd.DataFrame: Combined forecast table (all cities), or None on failure.
    """

    # -------------------------------------------------------------------------
    # STEP 1: Load and validate cleaned data
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("PART 3: CITY-SPECIFIC PROPHET FORECASTING")
    print("=" * 70)
    print("\nSTEP 1: Loading Data")
    print("-" * 50)

    if not os.path.exists(file_path):
        print(f"✗ File not found: {file_path}")
        print("  Run src/preprocess.py first.")
        return None

    df = pd.read_csv(file_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df[df['pm25'] > 0].copy()

    cities = sorted(df['city'].unique())
    print(f"✓ Loaded {len(df):,} rows  |  cities: {cities}")

    # -------------------------------------------------------------------------
    # STEP 2: Train one Prophet model per city and collect forecasts
    # -------------------------------------------------------------------------
    print("\nSTEP 2: Training city-specific Prophet models")
    print("-" * 50)

    all_forecasts = []   # Collect per-city forecast DataFrames
    all_alerts    = []   # Collect per-city alert records

    for city in cities:
        city_df = df[df['city'] == city].copy()

        # Train and forecast — function handles its own model saving
        future_forecast, last_ts = forecast_city_with_prophet(
            city_df, city, forecast_hours
        )

        # ---- Shape the forecast DataFrame ----
        # Keep only the four core columns and rename to match existing CSV schema
        fc = future_forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
        fc.columns = ['timestamp', 'pm25_predicted', 'pm25_lower', 'pm25_upper']
        for col in ['pm25_predicted', 'pm25_lower', 'pm25_upper']:
            fc[col] = fc[col].round(1)
        fc['city']        = city
        fc['alert_level'] = fc['pm25_predicted'].apply(assign_forecast_alert_level)

        all_forecasts.append(fc)

        # ---- Build forward-looking alerts for this city ----
        city_alerts = build_forecast_alerts(future_forecast, city, last_ts)
        all_alerts.extend(city_alerts)

    # -------------------------------------------------------------------------
    # STEP 3: Save combined forecast CSV
    # -------------------------------------------------------------------------
    print("\nSTEP 3: Saving combined forecast CSV")
    print("-" * 50)

    os.makedirs("outputs", exist_ok=True)

    combined_fc = pd.concat(all_forecasts, ignore_index=True)
    # Column order: city first so it's easy to filter in dashboards / pandas
    combined_fc = combined_fc[['city', 'timestamp', 'pm25_predicted',
                                'pm25_lower', 'pm25_upper', 'alert_level']]
    combined_fc.to_csv(FORECAST_CSV_OUTPUT, index=False)
    print(f"✓ Saved {len(combined_fc)} rows to {FORECAST_CSV_OUTPUT}")

    # -------------------------------------------------------------------------
    # STEP 4: Save forecast alerts CSV
    # -------------------------------------------------------------------------
    print("\nSTEP 4: Saving forecast alerts")
    print("-" * 50)

    if all_alerts:
        alerts_df = pd.DataFrame(all_alerts)
        alerts_df.to_csv(FORECAST_ALERTS_OUTPUT, index=False)
        print(f"✓ Saved {len(alerts_df)} forecast alert(s) to {FORECAST_ALERTS_OUTPUT}")

        print("\nForecast alerts by city:")
        for city, cnt in alerts_df['city'].value_counts().items():
            print(f"  {city:12}: {cnt} alert(s)")
    else:
        print("  No PM2.5 threshold breaches predicted in the next 24 hours — "
              "no forecast alerts generated.")
        # Write an empty file so downstream code can safely read it
        pd.DataFrame(columns=[
            'alert_type', 'city', 'forecast_timestamp', 'hours_ahead',
            'pm25_predicted', 'pm25_upper', 'aqi_level', 'alert_level',
            'affected_groups', 'protective_actions',
            'alert_text_en', 'alert_text_ur',
        ]).to_csv(FORECAST_ALERTS_OUTPUT, index=False)

    # -------------------------------------------------------------------------
    # STEP 5: Multi-city subplot figure (last 7 days history + 24h forecast)
    # -------------------------------------------------------------------------
    print("\nSTEP 5: Creating multi-city forecast plot")
    print("-" * 50)

    n_cities = len(cities)
    fig, axes = plt.subplots(n_cities, 1, figsize=(16, 5 * n_cities), sharex=False)
    if n_cities == 1:
        axes = [axes]

    for ax, city in zip(axes, cities):
        city_df    = df[df['city'] == city].copy().sort_values('timestamp')
        city_fc    = combined_fc[combined_fc['city'] == city].copy()

        # Show only the last 7 days of history to keep the chart readable
        recent_cutoff  = city_df['timestamp'].max() - pd.Timedelta(days=7)
        recent_history = city_df[city_df['timestamp'] >= recent_cutoff]

        # Historical line (blue)
        ax.plot(
            recent_history['timestamp'],
            recent_history['pm25'],
            color='steelblue', linewidth=1.2, alpha=0.8,
            label='Historical PM2.5', zorder=3
        )

        # Forecast line (green)
        ax.plot(
            city_fc['timestamp'],
            city_fc['pm25_predicted'],
            color='forestgreen', linewidth=2.0,
            label=f'Forecast ({forecast_hours}h)', zorder=4
        )

        # 95% confidence band (green, semi-transparent)
        # The band widens further into the future because uncertainty compounds
        # over time — this is honest and expected behaviour, not a flaw.
        ax.fill_between(
            city_fc['timestamp'],
            city_fc['pm25_lower'],
            city_fc['pm25_upper'],
            alpha=0.18, color='forestgreen',
            label='95% Confidence Interval'
        )

        # Hazardous threshold line (red dashed)
        ax.axhline(
            y=HAZARDOUS_THRESHOLD,
            color='red', linewidth=1.5, linestyle='--', alpha=0.7,
            label=f'Hazardous ({HAZARDOUS_THRESHOLD} µg/m³)'
        )

        # Unhealthy threshold line (orange dotted) — alert trigger level
        ax.axhline(
            y=UNHEALTHY_THRESHOLD,
            color='orange', linewidth=1.0, linestyle=':', alpha=0.7,
            label=f'Unhealthy ({UNHEALTHY_THRESHOLD} µg/m³)'
        )

        # Vertical divider: history | forecast boundary
        if not city_df.empty:
            boundary = city_df['timestamp'].max()
            ax.axvline(x=boundary, color='gray', linewidth=1.0,
                       linestyle=':', alpha=0.6)

        ax.set_title(f'{city}', fontsize=11, fontweight='bold')
        ax.set_ylabel('PM2.5 (µg/m³)', fontsize=9)
        ax.legend(loc='upper left', fontsize=7, framealpha=0.8, ncol=2)
        ax.grid(True, alpha=0.2)
        plt.setp(ax.get_xticklabels(), rotation=20, ha='right', fontsize=8)

    fig.suptitle(
        f'PM2.5 City-Specific Forecast — Next {forecast_hours} Hours  '
        f'(City-specific Prophet models)',
        fontsize=13, fontweight='bold', y=1.005
    )
    plt.tight_layout()
    plt.savefig(FORECAST_PLOT_OUTPUT, dpi=150, bbox_inches='tight')
    print(f"✓ Multi-city forecast plot saved: {FORECAST_PLOT_OUTPUT}")
    plt.close()

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    print("\nForecast summary per city:")
    for city in cities:
        city_fc = combined_fc[combined_fc['city'] == city]
        print(f"  {city:12}: min={city_fc['pm25_predicted'].min():6.1f}  "
              f"mean={city_fc['pm25_predicted'].mean():6.1f}  "
              f"max={city_fc['pm25_predicted'].max():6.1f} µg/m³  "
              f"| peak alert: {city_fc['alert_level'].iloc[city_fc['pm25_predicted'].values.argmax()]}")

    return combined_fc


# ============================================================================
# FUNCTION 18: Main function for Prophet Forecasting
# ============================================================================

def main_prophet_forecast():
    """
    Orchestrates the full city-specific Prophet forecasting pipeline as Part 3.
    """
    print("\n" + "=" * 70)
    print("PART 3: PROPHET — CITY-SPECIFIC 24-HOUR PM2.5 FORECAST")
    print("=" * 70)

    forecast_df = forecast_all_cities_with_prophet(INPUT_FILE, forecast_hours=FORECAST_HOURS)

    if forecast_df is None:
        print("✗ Forecasting failed — check that cleaned_data.csv exists.")
        return

    print("\n" + "=" * 70)
    print("✓ CITY-SPECIFIC PROPHET FORECASTING COMPLETE!")
    print("=" * 70)
    print(f"\nGenerated files:")
    print(f"  1. Forecast plot   : {FORECAST_PLOT_OUTPUT}")
    print(f"  2. Forecast CSV    : {FORECAST_CSV_OUTPUT}  (all 5 cities, 'city' column added)")
    print(f"  3. Forecast alerts : {FORECAST_ALERTS_OUTPUT}")
    print(f"  4. Saved models    : models/prophet_{{city}}.pkl  (one per city)")
    print(f"\nNext steps:")
    print(f"  • Open {FORECAST_PLOT_OUTPUT} to see the 5-city chart")
    print(f"  • Open {FORECAST_CSV_OUTPUT} to browse hourly predictions")
    print(f"  • Run dashboard:  streamlit run dashboard/app.py")


# ============================================================================
# MAIN EXECUTION FUNCTION
# ============================================================================

def main():
    """
    Main function that orchestrates the entire model training pipeline
    """
    print("\n" + "=" * 70)
    print("SmogAlert PK - Random Forest Model Training")
    print("=" * 70)

    # Step 1: Load and prepare data
    df = load_and_prepare_data(INPUT_FILE)
    if df is None:
        return

    # Step 2: Define features and target
    X, y, feature_names = define_features_and_target(df)

    # Step 3: Split data
    X_train, X_test, y_train, y_test = split_data(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )

    # Step 4: Train model
    model = train_random_forest(
        X_train, y_train,
        n_estimators=N_ESTIMATORS,
        random_state=RANDOM_STATE
    )

    # Step 5: Evaluate model
    accuracy, y_pred = evaluate_model(model, X_test, y_test)

    # Step 6: Plot confusion matrix
    plot_confusion_matrix(y_test, y_pred, CONFUSION_MATRIX_OUTPUT)

    # Step 7: Plot feature importance
    plot_feature_importance(model, feature_names, FEATURE_IMPORTANCE_OUTPUT)

    # Step 8: Save model
    save_model(model, MODEL_OUTPUT)

    # Final summary
    print("\n" + "=" * 70)
    print("✓ MODEL TRAINING COMPLETE!")
    print("=" * 70)
    print(f"\nModel Performance:")
    print(f"  Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"\nGenerated Files:")
    print(f"  1. Model: {MODEL_OUTPUT}")
    print(f"  2. Confusion Matrix: {CONFUSION_MATRIX_OUTPUT}")
    print(f"  3. Feature Importance: {FEATURE_IMPORTANCE_OUTPUT}")
    print(f"\nNext steps:")
    print(f"  1. Review the confusion matrix to see where the model struggles")
    print(f"  2. Check feature importance to understand what drives predictions")
    print(f"  3. Use the model: python src/alert_system.py")
    print(f"  4. Launch dashboard: streamlit run dashboard/app.py")

# ============================================================================
# Script Entry Point
# ============================================================================

if __name__ == "__main__":
    # Run PART 1: Random Forest Classification
    print("\n" + "=" * 70)
    print("STARTING MODEL TRAINING PIPELINE")
    print("=" * 70)
    print("\nThis script has two parts:")
    print("  PART 1: Random Forest - Predict AQI danger levels")
    print("  PART 2: Isolation Forest - Detect unusual PM2.5 spikes")
    print("=" * 70)

    main()

    # Run PART 2: Isolation Forest Anomaly Detection
    main_anomaly_detection()

    # Run PART 3: Prophet 24-Hour Forecast
    main_prophet_forecast()

    # Overall summary
    print("\n" + "=" * 70)
    print("✓✓✓ ALL MODEL TRAINING COMPLETE! ✓✓✓")
    print("=" * 70)
    print("\nYou now have:")
    print("  1. Random Forest Classifier - predicts AQI categories")
    print("  2. Isolation Forest         - detects anomalous readings")
    print("  3. Prophet Forecaster       - predicts next 24 hours of PM2.5")
    print("\nAll models, visualizations, and data files are ready!")
    print("Check the outputs/ folder for visualizations.")

