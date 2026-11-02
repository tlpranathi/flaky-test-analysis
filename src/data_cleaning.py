import os
import pandas as pd

from src.config import (
    FEATURES_PATH,
    RESULTS_PATH,
    RENAME,
    OUTPUT_DIR
)

# OUTPUT FILE

CLEANED_DATA = os.path.join(
    OUTPUT_DIR,
    "cleaned_data.csv"
)

# Columns that carry no information (leftover CSV index)

UNUSED_COLS = [
    "Unnamed: 0"
]

# 1. LOAD DATASETS

def load_datasets():
    """Load the raw features and results datasets."""

    print("\nLoading datasets...")

    features = pd.read_csv(FEATURES_PATH)
    results = pd.read_csv(RESULTS_PATH)

    print("Features:", features.shape)
    print("Results:", results.shape)

    return features, results

# 2. REMOVE UNUSED COLUMNS

def remove_unused_columns(df):
    """Drop leftover index columns that are not part of the data."""

    print("\n========== REMOVE UNUSED COLUMNS ==========")

    to_drop = [
        col for col in df.columns
        if col in UNUSED_COLS
        or col.startswith("Unnamed:")
    ]

    df = df.drop(columns=to_drop)

    print("Dropped columns:", to_drop if to_drop else "none")

    return df

# 3. RENAME COLUMNS

def rename_columns(df):
    """Rename feature columns using the team mapping in config.RENAME."""

    print("\n========== RENAME COLUMNS ==========")

    mapping = {
        old: new for old, new in RENAME.items()
        if old in df.columns
    }

    missing = [
        old for old in RENAME
        if old not in df.columns
    ]

    df = df.rename(columns=mapping)

    print("Renamed columns:", len(mapping))

    if missing:
        print("WARNING: Mapped columns not found:", missing)

    return df

# 4. SAVE CLEANED DATA

def save_cleaned_data(df, path=CLEANED_DATA):
    """Save the cleaned dataset to the outputs folder."""

    os.makedirs(os.path.dirname(path), exist_ok=True)

    df.to_csv(path, index=False)

    print("\nCleaned data saved to:", path)

# 5. MAIN EXECUTION

def main():

    features, results = load_datasets()

    rows_before, cols_before = features.shape

    features = remove_unused_columns(features)
    features = rename_columns(features)

    print("\n========== CLEANING SUMMARY ==========")

    print(f"Rows: {rows_before} -> {len(features)}")
    print(f"Columns: {cols_before} -> {len(features.columns)}")

    save_cleaned_data(features)


if __name__ == "__main__":
    main()
