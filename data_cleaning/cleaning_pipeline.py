import os
from processing_utils import *
import pandas as pd
import re


def dest_help(df):
    """Helper function to clean Destination column by filtering invalid entries"""
    print("Cleaning Destination column - filtering invalid entries...")
    # Ensure 'Destination' has at least one alphabetic character and is not just a country code
    df['Destination'] = df['Destination'].apply(
        lambda x: "NAN" if not re.search(r'[A-Za-z]', str(x)) or re.match(r'^[A-Z]{2}$', str(x)) else x
    )
    return df


def replace_with_key(df, column, name_variants):
    """Replace values in a column with standardized names using a mapping dictionary"""
    print(f"Standardizing {column} names using mapping dictionary...")
    df[column] = df[column].apply(lambda x: match_names(x, name_variants))
    return df


def main():
    file_path = '../data/cleaned_atr.csv'
    output_path = '../data/prepared.parquet'

    print(f"\n{'=' * 50}")
    print("STARTING DATA PREPARATION PIPELINE")
    print(f"{'=' * 50}\n")

    # ------ Load the file ------
    print(f"Step 1: Loading data from {file_path}")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Input file not found: {file_path}")

    df = pd.read_csv(file_path)
    print(f"Data loaded successfully. Shape: {df.shape}")
    print("Sample data:")
    print(df.head(3))

    # ------ Step 1: proper type conversion ------
    print("\nStep 2: Performing type conversions...")

    # String conversions
    df['Destination'] = df['Destination'].fillna("nan").astype('string')
    df['AisSourcen'] = df['AisSourcen'].fillna("nan").astype('string')

    # Datetime conversions
    df['StartTime'] = pd.to_datetime(df['StartTime'], utc=True)
    df['EndTime'] = pd.to_datetime(df['EndTime'], utc=True)
    df['time'] = pd.to_datetime(df['time'], utc=True)

    # Categorical conversions
    df['StartPort'] = df['StartPort'].astype('string').astype('category')
    df['EndPort'] = df['EndPort'].astype('string').astype('category')
    df['Destination'] = df['Destination'].astype('string').astype('category')

    print("\nData types after conversion:")
    print(df.dtypes)

    # ------ Step 2: Clean the 'Destination' column ------
    print("\nStep 3: Cleaning text columns...")

    # Convert all string columns to uppercase
    text_columns = df.select_dtypes(include=['string']).columns
    for col in text_columns:
        df[col] = df[col].str.upper()
        print(f"Converted {col} to uppercase")

    # Clean destination column
    print("Cleaning Destination column...")
    df['Destination'] = df['Destination'].apply(clean_destination)

    # Handle destinations with '>' character
    mask = df['Destination'].str.contains('>', na=False)
    print(f"Found {mask.sum()} entries with '>' in Destination")
    df.loc[mask, 'Destination'] = df.loc[mask, 'Destination'].str.split('>').str[1]

    # Standardize destination names
    df = replace_with_key(df, 'Destination', full_dict)

    # Handle destinations with '.' character
    mask = df['Destination'].str.contains('.', na=False)
    print(f"Found {mask.sum()} entries with '.' in Destination")
    df.loc[mask, 'Destination'] = df.loc[mask, 'Destination'].str.split('.').str[0]

    # Final destination cleaning
    df = dest_help(df)

    print("\nUnique Destination values after cleaning:")
    print(df['Destination'].value_counts().head(10))

    # ------ Step 3: Deal with missing values ------
    # print("\nStep 4: Handling missing values and duplicates...")
    print("\nStep 4: Droping duplicates...")

    # Remove duplicates
    initial_count = len(df)
    df = df.drop_duplicates()
    final_count = len(df)
    print(f"Removed {initial_count - final_count} duplicate rows")

    # Note now in noise handling

    # Drop unnecessary column

    # df.drop(columns=['AisSourcen'], inplace=True)
    # print("Dropped AisSourcen column")

    # Replace zeros with NaN for numerical columns
    # num_cols = ['Length', 'Breadth', 'Draught']
    # for col in num_cols:
    #     zero_count = (df[col] == 0).sum()
    #     df[col] = df[col].replace(0, np.nan)
    #     print(f"Replaced {zero_count} zeros with NaN in {col}")
    #
    # # Fill missing values with shiptype means
    # print("\nFilling missing values with shiptype means...")
    # for col in num_cols:
    #     missing_before = df[col].isna().sum()
    #     df[col] = df[col].fillna(df.groupby('shiptype')[col].transform('mean'))
    #     missing_after = df[col].isna().sum()
    #     print(f"Filled {missing_before - missing_after} missing values in {col}")

    # Fill missing destinations

    # print("\nFilling missing destinations using proximity method...")
    # missing_before = df['Destination'].isna().sum()
    # df = fill_missing_destinations_by_proximity(df)
    # missing_after = df['Destination'].isna().sum()
    # print(f"Filled {missing_before - missing_after} missing destinations")

    # Final duplicate check
    # df = df.drop_duplicates()
    # print(f"Final row count after all cleaning: {len(df)}")
    #
    # # Missing values report
    # print("\nMissing values percentage report:")
    # missing_report = (df.isnull().sum() / len(df) * 100).round(1)
    # print(missing_report)

    # ------ Step 4: Save the cleaned DataFrame ------
    print(f"\nStep 5: Saving cleaned data to {output_path}")
    df.to_parquet(output_path)
    print("Data saved successfully!")

    print(f"\n{'=' * 50}")
    print("DATA PREPARATION COMPLETE")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    main()
