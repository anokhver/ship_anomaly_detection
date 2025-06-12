import os
import pandas as pd
import numpy as np
from pathlib import Path

from data_cleaning.utils.fill_missing_utils import (
    get_percentage_missing,
    fill_missing,
    all_fill_with_mode,
    plot_missing,
    column_mapping,
    fill_missing_destinations_by_proximity, get_entries_with_missing_values
)
from data_cleaning.utils.utils import remove_duplicates


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

    # df['Destination'] = df['Destination'].fillna(pd.NA)

    return df


def handle_ship_dimensions(df: pd.DataFrame) -> pd.DataFrame:
    """Handle missing ship dimension values using mode filling and regression."""
    print("Processing ship dimensions...")

    # Fill basic dimensions with mode
    all_fill_with_mode(df, 'Length')
    all_fill_with_mode(df, 'Breadth')

    # Use regression to fill remaining missing values
    df = fill_missing(df, 'Length', ['Breadth', 'Draught'])
    df = fill_missing(df, 'Breadth', ['Length', 'Draught'])
    df = fill_missing(df, 'Draught', ['Length', 'Breadth'], round_values=False)

    return df


def handle_shiptype(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing shiptype values using forward fill within valid trips."""
    print("Processing ship types...")

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


def create_output_splits(df: pd.DataFrame, output_dir: str = '../data/manual_labeling/') -> None:
    """Create port-specific data splits for manual labeling."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

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
        file_path: str = 'data/2_destination_norm.parquet',
        output_path: str = '../data/fix_noise.parquet'
) -> None:
    """
    Main pipeline for cleaning noise in ship AIS data.

    Args:
        file_path: Input parquet file path
        output_path: Output parquet file path
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
    plot_dir = Path('plots')
    plot_dir.mkdir(exist_ok=True)

    plot = plot_missing(df)
    print(f"Missing values plot saved to {plot_dir / 'missing_values_1.png'}")
    plot.figure.savefig(plot_dir / 'missing_values.png', dpi=300, bbox_inches='tight')

    # Step 3: Fill missing destinations
    print("\nStep 3: Filling missing destinations")
    print(get_entries_with_missing_values(df, 'Destination'))
    df = fill_missing_destinations_by_proximity(df)
    print(f"Destination missing: {get_percentage_missing(df, 'Destination'):.2f}%")

    # Step 4: Handle ship characteristics
    print("\nStep 4: Processing ship characteristics forward fill")
    df = handle_shiptype(df)
    df = handle_draught(df)

    # Fill COG with 0 (common practice for missing course over ground)
    df['COG'] = df['COG'].fillna(0)

    # Step 5: Handle ship dimensions
    print("\n\nStep 5: Processing ship dimensions with all NAN entries for the trip")
    df = handle_ship_dimensions(df)

    # Verify completeness
    print("\nFinal missing value summary:")
    critical_columns = ['Draught', 'Length', 'Breadth', 'shiptype']
    for col in critical_columns:
        missing_pct = get_percentage_missing(df, col)
        print(f"{col}: {missing_pct:.2f}% missing")

    plot = plot_missing(df)
    print(f"Missing values plot saved to {plot_dir / 'missing_values_2.png'}")
    plot.figure.savefig(plot_dir / 'missing_values_2.png', dpi=300, bbox_inches='tight')

    df = remove_duplicates(df)

    # Step 6: Finalize dataset
    print("\nStep 6: Finalizing dataset")
    df_final = df.rename(columns=column_mapping)
    df_final['is_anomaly'] = None

    # Create output splits for manual labeling
    create_output_splits(df_final)

    # Save main output
    try:
        df.to_parquet(output_path)
        print(f"\nProcessed data saved to: {output_path}")
        print(f"Final dataset shape: {df.shape}")
    except Exception as e:
        raise RuntimeError(f"Failed to save output: {e}")


def main():
    """Entry point for the noise handling pipeline."""
    try:
        noise_handling()
        print("\nPipeline completed successfully!")
    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")
        raise


if __name__ == "__main__":
    main()