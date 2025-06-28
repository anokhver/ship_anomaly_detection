import pandas as pd


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate rows from the dataframe."""
    print("Removing duplicate rows...")

    initial_count = len(df)
    df = df.drop_duplicates()
    final_count = len(df)

    removed_count = initial_count - final_count
    if removed_count > 0:
        print(f"  - Removed {removed_count:,} duplicate rows ({removed_count / initial_count * 100:.1f}%)")
    else:
        print("  - No duplicates found")

    return df
