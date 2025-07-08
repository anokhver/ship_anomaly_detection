import os
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import r2_score

from utils.fill_missing_utils import (
    get_percentage_missing,
    all_fill_with_mode,
    plot_missing,
    column_mapping,
    fill_missing_destinations_by_proximity,
    get_entries_with_missing_values
)
from utils.utils import remove_duplicates


def convert_impossible_to_nan(df: pd.DataFrame) -> pd.DataFrame:
    """Convert physically impossible values to NaN for ship measurements."""
    df = df.copy()

    # Physical dimensions must be positive
    df['Length'] = df['Length'].apply(lambda x: x if x > 0 else np.nan)
    df['Breadth'] = df['Breadth'].apply(lambda x: x if x > 0 else np.nan)
    df['Draught'] = df['Draught'].apply(lambda x: x if x > 0 else np.nan)
    df['shiptype'] = df['shiptype'].apply(lambda x: x if x > 0 else np.nan)

    # Navigation values validation
    df['TH'] = df['TH'].apply(lambda x: x if (0 <= x <= 360) or x == 511 else np.nan)
    df['COG'] = df['COG'].apply(lambda x: x if 0 <= x <= 360 else np.nan)

    # Remove unrealistic speeds (>30 knots for most commercial vessels)
    initial_count = len(df)
    df = df[df['SOG'] < 30]
    removed_count = initial_count - len(df)

    if removed_count > 0:
        print(f"Removed {removed_count} rows with unrealistic SOG values")

    # Fill destination with pd.NA (from notebook)
    df['Destination'] = df['Destination'].fillna(pd.NA)

    return df


def fill_missing_regression(df: pd.DataFrame, target_col: str, feature_cols: list,
                            round_values: bool = True) -> pd.DataFrame:
    """
    Fill missing values using HistGradientBoostingRegressor.
    This function was in the notebook but missing from your py file.
    """
    df_copy = df.copy()
    known = df_copy.dropna(subset=[target_col])
    missing = df_copy[df[target_col].isna()]

    if len(missing) == 0:
        print(f"No missing values to fill for {target_col}")
        return df_copy

    X_train = known[feature_cols]
    y_train = known[target_col]
    X_test = missing[feature_cols]

    model = HistGradientBoostingRegressor()
    model.fit(X_train, y_train)
    predicted_values = model.predict(X_test)

    if round_values:
        predicted_values = np.round(predicted_values).astype(int)

    df_copy.loc[df_copy[target_col].isna(), target_col] = predicted_values
    print(f"R2 score for {target_col}:", r2_score(y_train, model.predict(X_train)))

    return df_copy


def handle_ship_dimensions(df: pd.DataFrame) -> pd.DataFrame:
    """Handle missing ship dimension values using mode filling and regression."""
    print("Processing ship dimensions...")

    # Fill basic dimensions with mode first
    all_fill_with_mode(df, 'Length')
    all_fill_with_mode(df, 'Breadth')

    # Check correlations (from notebook)
    print(f"Draught-Length correlation: {df['Draught'].corr(df['Length']):.3f}")
    print(f"Draught-Breadth correlation: {df['Draught'].corr(df['Breadth']):.3f}")
    print(f"Length-Breadth correlation: {df['Length'].corr(df['Breadth']):.3f}")

    # Use regression to fill remaining missing values (this was the main missing part)
    df = fill_missing_regression(df, 'Length', ['Breadth', 'Draught'])
    df = fill_missing_regression(df, 'Breadth', ['Length', 'Draught'])
    df = fill_missing_regression(df, 'Draught', ['Length', 'Breadth'], round_values=False)

    return df


def handle_shiptype(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing shiptype values using forward fill within valid trips."""
    print("Processing ship types...")

    # Ensure shiptype has positive values only (from notebook)
    df['shiptype'] = df['shiptype'].apply(lambda x: x if x > 0 else np.nan)

    # Identify trips that have at least one valid shiptype
    valid_trips = df.groupby('TripID')['shiptype'].transform(lambda x: x.notna().any())

    # Forward-fill only in trips with valid shiptype data
    df['shiptype'] = (
        df.groupby('TripID')['shiptype']
        .ffill()
        .where(valid_trips)
        .combine_first(df['shiptype'])
    )

    print(f"Shiptype missing: {get_percentage_missing(df, 'shiptype'):.2f}%")
    return df


def handle_draught(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing draught values using trip-level median imputation."""
    print("Processing draught values...")

    # Find trips with mixed draught data (some missing, some present)
    mixed_draught_trips = (
        df.groupby("TripID")["Draught"]
        .apply(lambda x: x.notna().any() and x.isna().any())
        .loc[lambda x: x]
        .index.tolist()
    )

    if mixed_draught_trips:
        trip_medians = df[df["TripID"].isin(mixed_draught_trips)].groupby("TripID")["Draught"].median()

        for trip_id in mixed_draught_trips:
            mask = (df["TripID"] == trip_id) & (df["Draught"].isna())
            df.loc[mask, "Draught"] = trip_medians[trip_id]

        print(f"Filled draught for {len(mixed_draught_trips)} trips using trip medians")

    return df


def create_output_splits(df: pd.DataFrame, output_dir: str) -> None:
    """Create port-specific data splits for manual labeling."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Print unique ports (from notebook)
    print("Start ports:", df['start_port'].unique())
    print("End ports:", df['end_port'].unique())

    port_splits = {
        'KIEL': df[df['start_port'] == 'KIEL'].copy().reset_index(drop=True),
        'BREMERHAVEN': df[df['start_port'] == 'BREMERHAVEN'].copy().reset_index(drop=True)
    }

    for port, port_df in port_splits.items():
        if not port_df.empty:
            output_file = Path(output_dir) / f'from_{port}.parquet'
            port_df.to_parquet(output_file)
            print(f"Saved {len(port_df)} records for port {port}")


def noise_handling(
        file_path: str = 'data_to_clean/2_destination_norm.parquet',
        output_path: str = '../visualization/clean_data/all_anomalies_combined.parquet',
        create_splits: bool = True,
        splits_dir: str = None
) -> None:
    """
    Main pipeline for cleaning noise in ship AIS data.

    Args:
        file_path: Input parquet file path
        output_path: Output parquet file path
        create_splits: Whether to create port-specific splits
        splits_dir: Directory for port splits (defaults to output_path parent + 'manual_labeling')
    """
    print(f"\n{'=' * 50}")
    print("STARTING DATA NOISE CLEANING PIPELINE")
    print(f"{'=' * 50}\n")

    # Load and validate input file
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Input file not found: {file_path}")

    try:
        df = pd.read_parquet(file_path)
        print(f"Loaded {len(df):,} records from {file_path}")
    except Exception as e:
        raise RuntimeError(f"Failed to load data: {e}")

    # Step 1: Remove unwanted columns and duplicates
    print("\nStep 1: Removing duplicates and unwanted columns")
    if 'AisSourcen' in df.columns:
        df.drop(columns=['AisSourcen'], inplace=True)
        print("Removed 'AisSourcen' column")

    # Remove duplicates
    df = remove_duplicates(df)

    # Step 2: Convert impossible values to NaN
    print("\nStep 2: Converting impossible values to NaN & dropping some invalid rows")
    df = convert_impossible_to_nan(df)

    # Generate and save missing values plot
    plot_dir = Path(output_path).parent / 'plots'
    plot_dir.mkdir(exist_ok=True)

    # Step 3: Fill missing destinations
    print("\nStep 3: Filling missing destinations")
    print("Entries with missing destinations:", get_entries_with_missing_values(df, 'Destination'))
    df = fill_missing_destinations_by_proximity(df)
    print(f"Destination missing: {get_percentage_missing(df, 'Destination'):.2f}%")

    # Step 4: Handle ship characteristics
    print("\nStep 4: Processing ship characteristics forward fill")
    df = handle_shiptype(df)
    df = handle_draught(df)

    # Fill COG with 0 (common practice for missing course over ground)
    df['COG'] = df['COG'].fillna(0)
    print("Filled remaining COG values with 0")

    # Step 5: Handle ship dimensions (this was the main missing regression part)
    print("\nStep 5: Processing ship dimensions with regression for trips with all NaN entries")
    df = handle_ship_dimensions(df)

    # Verify completeness
    print("\nFinal missing value summary:")
    critical_columns = ['Draught', 'Length', 'Breadth', 'shiptype']
    for col in critical_columns:
        missing_pct = get_percentage_missing(df, col)
        print(f"{col}: {missing_pct:.2f}% missing")

    # Final duplicate removal
    df = remove_duplicates(df)

    # Step 6: Finalize dataset
    print("\nStep 6: Finalizing dataset")
    df_final = df.rename(columns=column_mapping)
    
    for col in df_final.columns:
        if pd.api.types.is_numeric_dtype(df_final[col]):
            df_final[col] = df_final[col].fillna(0)
        else:
            if pd.api.types.is_categorical_dtype(df_final[col]):
                if 'null' not in df_final[col].cat.categories:
                    df_final[col] = df_final[col].cat.add_categories(['null'])
            df_final[col] = df_final[col].fillna('null')

    df_final['is_anomaly'] = None

    # Create output splits for manual labeling if requested
    if create_splits:
        if splits_dir is None:
            splits_dir = Path(output_path).parent / 'manual_labeling'
        create_output_splits(df_final, splits_dir)

    # Save main output
    try:
        df_final.to_parquet(output_path)
        print(f"\nProcessed data saved to: {output_path}")
        print(f"Final dataset shape: {df_final.shape}")
    except Exception as e:
        raise RuntimeError(f"Failed to save output: {e}")


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Clean noise in ship AIS data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        'input_file',
        type=str,
        default='data/2_destination_norm.parquet',
        help='Path to input parquet file'
    )

    parser.add_argument(
        'output_file',
        type=str,
        default='../data/fix_noise.parquet',
        help='Path to output parquet file'
    )

    parser.add_argument(
        '--no-splits',
        action='store_true',
        help='Skip creating port-specific data splits'
    )

    parser.add_argument(
        '--splits-dir',
        type=str,
        default=None,
        help='Directory for port-specific splits (default: output_dir/manual_labeling)'
    )

    return parser.parse_args()


def main():
    """Entry point for the noise handling pipeline."""
    args = parse_arguments()

    try:
        noise_handling(
            file_path=args.input_file,
            output_path=args.output_file,
            create_splits=not args.no_splits,
            splits_dir=args.splits_dir
        )
        print("\nPipeline completed successfully!")
    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")
        raise


if __name__ == "__main__":
    main()