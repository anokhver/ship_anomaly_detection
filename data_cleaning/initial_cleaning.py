import os
import pandas as pd

from data_cleaning.utils.utils import remove_duplicates


def ensure_dir(path):
    """Create directory if it doesn't exist."""
    if not os.path.exists(path):
        os.makedirs(path)


def load_data(file_path):
    """Load CSV data with error handling."""
    print(f"Loading data from {file_path}")

    df = pd.read_csv(
        file_path,
        engine='python',
        on_bad_lines='skip',
        na_values=['?', '']
    )

    print(f"Loaded {len(df):,} rows and {len(df.columns)} columns")
    return df


def clean_time_column(df):
    """Clean and process time column with timezone handling."""
    if 'time' not in df.columns:
        print("Warning: No 'time' column found")
        return df

    print("Processing time column...")

    # Clear time (Problem with .KIEL timezone)
    df['time'] = (
        df['time'].astype(str)
        .str.replace(r"\..*$", "", regex=True)
        .str.strip("'")
    )

    # Convert to datetime
    df['time'] = pd.to_datetime(df['time'], format='mixed', errors='coerce')

    # Handle timezone conversion
    df['time'] = df['time'].dt.tz_localize('UTC', ambiguous='NaT', nonexistent='shift_forward')
    df['time'] = df['time'].dt.tz_convert('Europe/Berlin')

    print(f"Time column processed: {df['time'].notna().sum():,} valid timestamps")
    return df


def convert_numeric_columns(df, columns):
    """Convert specified columns to numeric."""
    print("Converting numeric columns...")

    for col in columns:
        if col in df.columns:
            original_count = df[col].notna().sum()
            df[col] = pd.to_numeric(df[col], errors='coerce')
            new_count = df[col].notna().sum()
            print(f"  {col}: {new_count:,} valid values ({original_count - new_count:,} lost)")
        else:
            print(f"  Warning: Column '{col}' not found")

    return df


def drop_columns(df, columns_to_drop):
    """Drop specified columns from dataframe."""
    if columns_to_drop is None:
        print("Warning: No columns specified to drop")
        return df

    existing_columns = [col for col in columns_to_drop if col in df.columns]
    missing_columns = [col for col in columns_to_drop if col not in df.columns]

    if existing_columns:
        df = df.drop(columns=existing_columns)
        print(f"Dropped columns: {existing_columns}")

    if missing_columns:
        print(f"Columns not found: {missing_columns}")

    return df


def save_cleaned_data(df, output_path):
    """Save cleaned dataframe to CSV."""
    ensure_dir(os.path.dirname(output_path) or '.')
    df.to_parquet(output_path, index=False)
    print(f"Cleaned data saved to {output_path}")
    print(f"Final dataset: {len(df):,} rows and {len(df.columns)} columns")


def initial_data_processing(
        input_path='../data/raw_data/merged.csv',
        columns_to_drop=None,
        output_cleaned='data/cleaned_atr.parquet'
):
    if columns_to_drop is None:
        columns_to_drop = ['ID', 'Name', 'Callsign', 'MMSI']

    """Main data cleaning pipeline."""
    print(f"\n{'=' * 50}")
    print("STARTING INITIAL CLEANING PIPELINE")
    print(f"{'=' * 50}\n")

    # Load data
    df = load_data(input_path)

    # Clean time column
    df = clean_time_column(df)

    # Convert numeric columns
    numeric_columns = ['Draught', 'TH', 'SOG', 'COG']
    df = convert_numeric_columns(df, numeric_columns)

    # Drop unwanted columns
    df_clean = drop_columns(df, columns_to_drop)

    # Remove duplicates
    df_clean = remove_duplicates(df_clean)

    # Save cleaned data
    save_cleaned_data(df_clean, output_cleaned)


def main():
    """Run the initial data processing."""
    initial_data_processing()


if __name__ == "__main__":
    main()
