import os
import re
from pathlib import Path
from typing import List, Optional

import pandas as pd

from data_cleaning.utils.normalization_utils import match_names, clean_destination
from data_cleaning.utils.utils import remove_duplicates


def convert_data_types(df: pd.DataFrame) -> pd.DataFrame:
    """Convert columns to appropriate data types for memory efficiency and processing."""
    print("Converting data types...")

    df = df.copy()

    # String conversions with proper null handling
    string_columns = ['Destination', 'AisSourcen', 'StartPort', 'EndPort']
    for col in string_columns:
        if col in df.columns:
            df[col] = df[col].fillna("nan").astype('string')

    # Datetime conversions with error handling
    datetime_columns = ['StartTime', 'EndTime', 'time']
    for col in datetime_columns:
        if col in df.columns:
            try:
                df[col] = pd.to_datetime(df[col], utc=True)
            except Exception as e:
                print(f"Warning: Could not convert {col} to datetime: {e}")

    # Categorical conversions for memory efficiency
    categorical_columns = ['StartPort', 'EndPort', 'Destination']
    for col in categorical_columns:
        if col in df.columns:
            df[col] = df[col].astype('category')

    # Handle shiptype separately due to special conversion needs
    if 'shiptype' in df.columns:
        try:
            df['shiptype'] = pd.to_numeric(df['shiptype'], errors='coerce').astype('Int64')
            df['shiptype'] = df['shiptype'].astype('category')
        except Exception as e:
            print(f"Warning: Could not convert shiptype: {e}")

    return df


def standardize_text_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Convert text columns to uppercase for consistency."""
    print("Standardizing text columns to uppercase...")

    df = df.copy()
    text_columns = df.select_dtypes(include=['string', 'object']).columns

    for col in text_columns:
        if df[col].dtype == 'string':
            df[col] = df[col].str.upper()
            print(f"  - Converted {col} to uppercase")

    return df


def clean_destination_column(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and standardize destination column values."""
    print("Processing destination column...")

    if 'Destination' not in df.columns:
        print("Warning: Destination column not found")
        return df

    df = df.copy()

    # Initial cleaning using utility function
    df['Destination'] = df['Destination'].apply(clean_destination)

    # Handle destinations with '>' character (route indicators)
    route_mask = df['Destination'].str.contains('>', na=False)
    if route_mask.sum() > 0:
        print(f"  - Processing {route_mask.sum()} entries with route indicators ('>')")
        df.loc[route_mask, 'Destination'] = (
            df.loc[route_mask, 'Destination']
            .str.split('>')
            .str[-1]  # Take the last part (final destination)
            .str.strip()
        )

    return df


def standardize_destination_names(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize destination names using mapping dictionary."""
    print("Standardizing destination names...")

    if 'Destination' not in df.columns:
        return df

    df = df.copy()
    initial_unique = df['Destination'].nunique()

    df['Destination'] = df['Destination'].apply(match_names)

    final_unique = df['Destination'].nunique()
    print(f"  - Reduced unique destinations from {initial_unique} to {final_unique}")

    return df


def filter_invalid_destinations(df: pd.DataFrame) -> pd.DataFrame:
    """Filter out invalid destination entries."""
    print("Filtering invalid destination entries...")

    if 'Destination' not in df.columns:
        return df

    df = df.copy()
    initial_count = len(df)

    # Mark invalid destinations as "NAN"
    df['Destination'] = df['Destination'].apply(
        lambda x: "NAN" if (
                not re.search(r'[A-Za-z]', str(x)) or  # No alphabetic characters
                re.match(r'^[A-Z]{2}$', str(x)) or  # Just country codes
                str(x).lower() in ['nan', 'none', '']  # Null-like values
        ) else x
    )

    invalid_count = (df['Destination'] == "NAN").sum()
    if invalid_count > 0:
        print(f"  - Marked {invalid_count} invalid destinations as 'NAN'")

    return df


def display_summary_statistics(df: pd.DataFrame) -> None:
    """Display summary statistics about the processed data."""
    print("\nData summary after processing:")
    print(f"  - Total records: {len(df):,}")
    print(f"  - Columns: {len(df.columns)}")
    print(f"  - Memory usage: {df.memory_usage(deep=True).sum() / 1024 ** 2:.1f} MB")

    if 'Destination' in df.columns:
        print(f"\nTop 10 destinations:")
        dest_counts = df['Destination'].value_counts().head(10)
        for dest, count in dest_counts.items():
            print(f"  - {dest}: {count:,} ({count / len(df) * 100:.1f}%)")


def normalization(
        file_path: str = 'data/cleaned_atr.parquet',
        output_path: str = 'data/2_destination_norm.parquet'
) -> None:
    """
    Main pipeline for normalizing ship AIS data.

    Args:
        file_path: Input CSV file path
        output_path: Output parquet file path
    """
    print(f"\n{'=' * 50}")
    print("STARTING DATA NORMALIZATION PIPELINE")
    print(f"{'=' * 50}\n")

    # Validate input file
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Input file not found: {file_path}")

    # Load data
    print(f"Step 1: Loading data from {file_path}")
    try:
        df = pd.read_parquet(file_path)
        print(f"  - Loaded {len(df):,} records with {len(df.columns)} columns")
    except Exception as e:
        raise RuntimeError(f"Failed to load data: {e}")

    # Data type conversions
    print(f"\nStep 2: Converting data types")
    df = convert_data_types(df)

    # Text standardization
    print(f"\nStep 3: Standardizing text columns")
    df = standardize_text_columns(df)

    # Destination processing
    print(f"\nStep 4: Processing destination data")
    df = clean_destination_column(df)
    df = standardize_destination_names(df)
    df = filter_invalid_destinations(df)

    # Remove duplicates
    print(f"\nStep 5: Removing duplicates")
    df = remove_duplicates(df)

    # Display summary
    display_summary_statistics(df)

    # Save processed data
    print(f"\nStep 6: Saving processed data")
    try:
        # Ensure output directory exists
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)

        df.to_parquet(output_path, index=False)
        print(f"  - Data saved to: {output_path}")
        print(f"  - Final shape: {df.shape}")

        # Verify file was created and get size
        if os.path.exists(output_path):
            file_size = os.path.getsize(output_path) / 1024 ** 2
            print(f"  - File size: {file_size:.1f} MB")

    except Exception as e:
        raise RuntimeError(f"Failed to save data: {e}")


def main():
    """Entry point for the normalization pipeline."""
    try:
        normalization()
        print("\nPipeline completed successfully!")
    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")
        raise


if __name__ == "__main__":
    main()
