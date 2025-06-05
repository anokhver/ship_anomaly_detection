import pandas as pd
import os
from dest_normalization import *

file_path = '../models/cleaned_atr.csv'
output_path = '../data/prepared_data.parquet'

if not os.path.exists(file_path):
    print(f"File not found: {file_path}")
    # Load the file if it exists
df = pd.read_csv(file_path)
df.head()

df['Destination'] = df['Destination'].fillna("nan").astype('string')
df['AisSourcen'] = df['AisSourcen'].fillna("nan").astype('string')

# Convert time columns to datetime with timezone awareness
df['StartTime'] = pd.to_datetime(df['StartTime'], utc=True)
df['EndTime'] = pd.to_datetime(df['EndTime'], utc=True)
df['time'] = pd.to_datetime(df['time'], utc=True)  # Appears to have timezone info (+01:00)

df['StartPort'] = df['StartPort'].astype('string').astype('category')
df['EndPort'] = df['EndPort'].astype('string').astype('category')
print(df.dtypes)

text_columns = df.select_dtypes(include=['string']).columns
for col in text_columns:
    df[col] = df[col].str.upper()  # Ensure string type and uppercase

df['Destination'] = df['Destination'].where(
    df['Destination'].str.contains(r'[A-Za-z]', na=False),
    "NAN"
)  # Atle ast one alphabetic

df['Destination'] = df['Destination'].apply(
    lambda x: "NAN" if re.match(r'^[A-Z]{2}$', str(x)) else x
)  # only country code

df['Destination'] = df['Destination'].apply(clean_destination)

mask = df['Destination'].str.contains('>', na=False)
df.loc[mask, 'Destination'] = df.loc[mask, 'Destination'].str.split('>').str[1]

text_columns = df.select_dtypes(include=['string']).columns
for col in text_columns:
    df[col] = df[col].str.upper()  # Ensure string type and uppercase

df['Destination'] = df['Destination'].where(
    df['Destination'].str.contains(r'[A-Za-z]', na=False),
    "NAN"
)  # Atle ast one alphabetic

df['Destination'] = df['Destination'].apply(
    lambda x: "NAN" if re.match(r'^[A-Z]{2}$', str(x)) else x
)  # only country code

df['Destination'] = df['Destination'].apply(clean_destination)

mask = df['Destination'].str.contains('>', na=False)
df.loc[mask, 'Destination'] = df.loc[mask, 'Destination'].str.split('>').str[1]


def replace_with_key(df, column, name_variants):
    df[column] = df[column].apply(lambda x: match_names(x, name_variants))
    return df

df[['Destination']].reset_index().drop_duplicates(subset=['Destination'])
# mask = df['Destination'].str.contains('.', na=False)
# df.loc[mask, 'Destination'] = df.loc[mask, 'Destination'].str.split('.').str[0]
# mask = df['Destination'].str.contains('/', na=False)
# df.loc[mask, 'Destination'] = df.loc[mask, 'Destination'].str.split('/').str[0]

df['Destination'] = df['Destination'].astype('category')

df = df.drop_duplicates()
df.drop(columns=['AisSourcen'], inplace=True)  # Drop the 'AisSource' column as it is not needed really
df['Draught'] = df['Draught'].fillna(df['Draught'].median())
df['Destination'] = df.groupby('TripID')['Destination'].transform(lambda x: x.ffill().bfill())

df.to_parquet(output_path)
