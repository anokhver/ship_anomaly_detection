import os
import pandas as pd


def ensure_dir(path):
    """Create directory if it doesn't exist."""
    if not os.path.exists(path):
        os.makedirs(path)


def load_data_files(file_paths):
    """Load multiple CSV files with error handling."""
    dfs = []
    for file_path in file_paths:
        print(f"Loading data from {file_path}")
        df = pd.read_csv(
            file_path,
            engine='python',
            on_bad_lines='skip',
            na_values=['', '?']
        )
        print(f"Loaded {len(df):,} rows and {len(df.columns)} columns")
        dfs.append(df)
    return dfs


def check_mixed_types_and_examples(df):
    """Check for mixed types in dataframe columns and display examples."""
    mixed = []
    for column in df.columns:
        unique_types = df[column].apply(type).unique()
        if len(unique_types) > 1:
            print(f"Column '{column}' has mixed types: {unique_types}")
            # Display examples of string and float values
            string_values = df[column][df[column].apply(type) == str].head()
            float_values = df[column][df[column].apply(type) == float].head()
            print(f"Examples of string values in '{column}':\n{string_values}")
            print(f"Examples of float values in '{column}':\n{float_values}")
            mixed.append(column)
    return mixed


def clean_and_type_dataframes(dfs, columns_to_drop):
    """Clean and type convert dataframes."""
    print("Processing and typing dataframes...")
    dfs_clean = []

    for df in dfs:
        print(f"Processing dataframe with {len(df):,} rows")

        # Drop unwanted columns
        df_clean = df.drop(columns=columns_to_drop, errors='ignore')
        dropped_cols = [col for col in columns_to_drop if col in df.columns]
        if dropped_cols:
            print(f"Dropped columns: {dropped_cols}")

        # Convert datetime columns
        datetime_columns = ['StartTime', 'EndTime', 'time']
        for col in datetime_columns:
            if col in df_clean.columns:
                original_count = df_clean[col].notna().sum()
                df_clean[col] = pd.to_datetime(df_clean[col], utc=True)
                new_count = df_clean[col].notna().sum()
                print(f" {col}: {new_count:,} valid timestamps ({original_count - new_count:,} lost)")

        # Convert categorical columns
        categorical_columns = ['StartPort', 'EndPort', 'Destination']
        for col in categorical_columns:
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].astype('string').astype('category')
                print(f" {col}: converted to category")

        df_clean['shiptype'] = df_clean['shiptype'].astype('int').astype('category')

        dfs_clean.append(df_clean)

    return dfs_clean


def merge_dataframes(dfs_clean):
    """Merge cleaned dataframes."""
    print("Merging dataframes...")
    df_final = pd.concat(dfs_clean, ignore_index=True)
    print(f"Final merged dataset: {len(df_final):,} rows and {len(df_final.columns)} columns")
    return df_final


def sort_and_group_data(df):
    """Sort data by time and group by TripID and StartPort."""
    print("Sorting data by time...")
    df_sorted = df.sort_values('time')
    print("Data sorted successfully")

    # Group by TripID and StartPort (for verification)
    grouped = df_sorted.groupby(['TripID', 'StartPort'])
    print(f"Data grouped into {len(grouped)} groups")

    return df_sorted


def save_merged_data(df, output_path):
    """Save merged dataframe to parquet."""
    ensure_dir(os.path.dirname(output_path) or '.')
    df.to_parquet(output_path)
    print(f"Merged data saved to {output_path}")
    print(f"Final dataset: {len(df):,} rows and {len(df.columns)} columns")


def merge_and_process_data(
        input_files=None,
        columns_to_drop=None,
        output_file='data/1_merged_typed_data.parquet'
):
    """Main data merging and processing pipeline."""
    if input_files is None:
        input_files = ["../data/raw_data/b-h.csv", "../data/raw_data/k-g.csv"]

    if columns_to_drop is None:
        columns_to_drop = ['ID', 'Name', 'Callsign', 'MMSI', 'AisSourcen']

    print(f"\n{'=' * 50}")
    print("STARTING DATA MERGE AND PROCESSING PIPELINE")
    print(f"{'=' * 50}\n")

    # Load data files
    dfs = load_data_files(input_files)

    # Clean and type convert dataframes
    dfs_clean = clean_and_type_dataframes(dfs, columns_to_drop)

    # Check for mixed types in first dataframe
    if dfs_clean:
        print("\nChecking for mixed types in first dataframe:")
        mixed_cols = check_mixed_types_and_examples(dfs_clean[0])
        if not mixed_cols:
            print("No mixed types found")

    # Merge dataframes
    df_final = merge_dataframes(dfs_clean)

    # Display info about final dataframe
    print("\nFinal dataframe info:")
    df_final.info()

    # Sort and group data
    df_final = sort_and_group_data(df_final)

    # Save merged data
    save_merged_data(df_final, output_file)


def main():
    """Run the data merge and processing."""
    import argparse

    parser = argparse.ArgumentParser(description='Merge and process ship tracking data')
    parser.add_argument('--input-files', nargs='+',
                        default=None,
                        help='Input CSV files to merge')
    parser.add_argument('--columns-to-drop', nargs='+',
                        default=None,
                        help='Columns to drop from the data')
    parser.add_argument('--output-file',
                        default=None,
                        help='Output parquet file path')

    args = parser.parse_args()

    merge_and_process_data(
        input_files=args.input_files,
        columns_to_drop=args.columns_to_drop,
        output_file=args.output_file
    )


if __name__ == "__main__":
    main()