import argparse
import os
import re
import sys
from pathlib import Path
from typing import Dict, Any

import pandas as pd

from data_cleaning.utils.normalization_utils import clean_destination, match_names


def check_unique_values(df: pd.DataFrame) -> Dict[str, int]:
    """Check unique values in each column of the DataFrame."""
    print("Checking unique values in each column...")

    col_un = {}
    for col in df.columns:
        clean_series = df[col].dropna()
        nunique = clean_series.nunique()
        col_un[col] = nunique
        print(f"  - {col}: {nunique} unique values")

    return col_un


def standardize_text_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Convert text columns to uppercase for consistency."""
    print("Standardizing text columns to uppercase...")

    df = df.copy()
    text_columns = df.select_dtypes(include=['string']).columns

    for col in text_columns:
        df[col] = df[col].str.upper()
        print(f"  - Converted {col} to uppercase")

    return df


def filter_invalid_destinations(df: pd.DataFrame) -> pd.DataFrame:
    """Filter out invalid destination entries."""
    print("Filtering invalid destination entries...")

    if 'Destination' not in df.columns:
        print("Warning: Destination column not found")
        return df

    df = df.copy()
    initial_count = len(df)

    # Mark destinations without alphabetic characters or just country codes as NaN
    df['Destination'] = df['Destination'].apply(
        lambda x: pd.NA if (
                not re.search(r'[A-Za-z]', str(x)) or
                re.match(r'^[A-Z]{2}$', str(x))
        ) else x
    )

    invalid_count = df['Destination'].isna().sum()
    print(f"  - Marked {invalid_count} invalid destinations as NaN")

    return df


def find_values_with_special_chars(df: pd.DataFrame) -> list:
    """Find values with special characters in the 'Destination' column."""
    return [
        value for value in df['Destination'].unique()
        if re.search(r'[^A-Za-z0-9]', str(value)) and pd.notna(value)
    ]


def clean_destination_column(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and standardize destination column values."""
    print("Processing destination column...")

    if 'Destination' not in df.columns:
        print("Warning: Destination column not found")
        return df

    df = df.copy()

    # Show initial special characters count
    dest_before = find_values_with_special_chars(df)
    print(f"  - Found {len(dest_before)} values with special characters before cleaning")

    # Clean all destination data using utility function
    df['Destination'] = df['Destination'].apply(clean_destination)

    # Handle destinations with '>' character (route indicators)
    route_mask = df['Destination'].str.contains('>', na=False)
    if route_mask.sum() > 0:
        print(f"  - Processing {route_mask.sum()} entries with route indicators ('>')")
        # Take the part after '>' as the final destination
        df.loc[route_mask, 'Destination'] = (
            df.loc[route_mask, 'Destination']
            .str.split('>')
            .str[1]  # Take the second part (final destination)
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

    # Apply name matching/standardization
    df['Destination'] = df['Destination'].apply(match_names)

    final_unique = df['Destination'].nunique()
    print(f"  - Reduced unique destinations from {initial_unique} to {final_unique}")

    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate rows from the DataFrame."""
    print("Removing duplicate rows...")

    initial_count = len(df)
    df = df.drop_duplicates()
    final_count = len(df)

    removed_count = initial_count - final_count
    print(f"  - Removed {removed_count} duplicate rows")
    print(f"  - Remaining records: {final_count:,}")

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


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments for the destination normalization pipeline."""
    parser = argparse.ArgumentParser(
        description="Destination Normalization Pipeline for Ship AIS Data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s
  %(prog)s --input data/raw_data.parquet --output data/processed.parquet
  %(prog)s -i data/input.parquet -o data/output.parquet --verbose
  %(prog)s --input data/test.parquet --skip-duplicates --no-summary
        """
    )

    # Input/Output arguments
    parser.add_argument(
        '-i', '--input',
        type=str,
        default='../data/1_merged_typed_data.parquet',
        help='Input parquet file path (default: %(default)s)'
    )

    parser.add_argument(
        '-o', '--output',
        type=str,
        default='../data/2_destination_norm.parquet',
        help='Output parquet file path (default: %(default)s)'
    )
    # Output options
    parser.add_argument(
        '--no-summary',
        action='store_true',
        help='Skip displaying summary statistics'
    )

    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose output with detailed processing information'
    )

    return parser.parse_args()


def validate_arguments(args: argparse.Namespace) -> None:
    """Validate command line arguments and input conditions."""
    # Check input file exists
    if not os.path.exists(args.input):
        print(f"❌ Error: Input file does not exist: {args.input}")
        sys.exit(1)

    # Check input file extension
    if not args.input.lower().endswith('.parquet'):
        print(f"⚠️  Warning: Input file doesn't have .parquet extension: {args.input}")

    # Check output file existence
    if os.path.exists(args.output) and not args.force:
        response = input(f"Output file exists: {args.output}\nOverwrite? (y/N): ")
        if response.lower() not in ['y', 'yes']:
            print("Operation cancelled.")
            sys.exit(0)

    # Validate output directory
    output_dir = Path(args.output).parent
    if not output_dir.exists():
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            print(f"Created output directory: {output_dir}")
        except PermissionError:
            print(f"❌ Error: Cannot create output directory: {output_dir}")
            sys.exit(1)

    # Check file permissions
    try:
        with open(args.input, 'rb') as f:
            pass
    except PermissionError:
        print(f"❌ Error: No read permission for input file: {args.input}")
        sys.exit(1)


def destination_normalization(
        file_path: str = 'data/1_merged_typed_data.parquet',
        output_path: str = 'data/2_destination_norm.parquet',
        verbose: bool = False,
        show_summary: bool = True,
) -> None:
    """
    Main pipeline for normalizing destination data in ship AIS records.

    This pipeline processes destination names to standardize formats,
    remove inconsistencies, and clean invalid entries.

    Args:
        file_path: Input parquet file path
        output_path: Output parquet file path
        skip_text_norm: Skip text column standardization
        skip_invalid_filter: Skip invalid destination filtering
        skip_duplicates: Skip duplicate removal
        skip_name_standardization: Skip destination name standardization
        show_summary: Display summary statistics
        verbose: Enable verbose output
        dry_run: Run without saving output
        validate_input: Perform additional input validation
    """
    print(f"\n{'=' * 50}")
    print("STARTING DESTINATION NORMALIZATION PIPELINE")
    print(f"{'=' * 50}\n")

    if verbose:
        print("Pipeline configuration:")
        print(f"  - Input file: {file_path}")
        print(f"  - Output file: {output_path}")
        print(f"  - Show summary: {show_summary}")
        print()

    # Validate input file
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Input file not found: {file_path}")

    # Load data
    print(f"Step 1: Loading data from {file_path}")
    try:
        df = pd.read_parquet(file_path)
        print(f"  - Loaded {len(df):,} records with {len(df.columns)} columns")
        if verbose:
            print(f"  - Data types: {df.dtypes.value_counts().to_dict()}")
            print(f"  - Memory usage: {df.memory_usage(deep=True).sum() / 1024 ** 2:.1f} MB")
    except Exception as e:
        raise RuntimeError(f"Failed to load data: {e}")

    # Check initial unique values
    print(f"\nStep 2: Analyzing initial data structure")
    unique_values_before = check_unique_values(df) if verbose else {}

    # Text standardization
    print(f"\nStep 3: Standardizing text columns")
    df = standardize_text_columns(df)

    # Check changes after case normalization
    if verbose:
        unique_values_after = check_unique_values(df)
        changed_columns = [
            col for col in df.columns
            if unique_values_after[col] != unique_values_before[col]
        ]

        if changed_columns:
            print(f"  - Case normalization changed unique values in: {changed_columns}")
    # Destination processing
    print(f"\nStep 4: Processing destination data")

    # Show sample destinations before processing
    if verbose and 'Destination' in df.columns:
        sample_destinations = df['Destination'].dropna().unique()[:10]
        print(f"  - Sample destinations before processing: {sample_destinations.tolist()}")

    # Filter invalid destinations first
    df = filter_invalid_destinations(df)

    # Clean destination column
    df = clean_destination_column(df)

    # Standardize destination names
    df = standardize_destination_names(df)

    # Show final unique destinations
    if verbose and 'Destination' in df.columns:
        final_destinations = df['Destination'].dropna().unique()
        print(f"  - Final unique destinations: {len(final_destinations)}")
        print(f"  - Sample final destinations: {final_destinations[:10].tolist()}")

    # Remove duplicates
    print(f"\nStep 5: Removing duplicates")
    df = remove_duplicates(df)

    # Display summary
    if show_summary:
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
    """Entry point for the destination normalization pipeline."""
    try:
        # Parse command line arguments
        args = parse_arguments()

        # Validate arguments
        validate_arguments(args)

        # Run the pipeline with parsed arguments
        destination_normalization(
            file_path=args.input,
            output_path=args.output,
            show_summary=not args.no_summary,
            verbose=args.verbose,
        )

        print("\nDestination normalization pipeline completed successfully!")

    except KeyboardInterrupt:
        print("\n\nPipeline interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")

        if args.verbose if 'args' in locals() else False:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
