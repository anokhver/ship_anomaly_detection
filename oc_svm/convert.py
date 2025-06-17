import pandas as pd

def csv_to_parquet(input_csv: str, output_parquet: str) -> None:
    """
    Konwertuje plik CSV na Apache Parquet, obsługując brakujące wartości
    w kolumnach całkowitoliczbowych dzięki typom nullable Int64.
    """
    df = pd.read_csv(
        input_csv,
        sep=';',                # średnik jako separator
        decimal=',',            # przecinek jako separator dziesiętny
        dayfirst=True,          # format daty D.M.Y
        parse_dates=['start_time', 'end_time', 'time_stamp'],
        dtype={
            'trip_id':           'Int64',   # nullable int
            'ship_type':         'Int64',
            'length':            'float',
            'breadth':           'float',
            'draught':           'float',
            'speed_over_ground': 'float',
            'course_over_ground':'float',
            'true_heading':      'float',
            'is_anomaly':        'boolean'  # pandas nullable bool
        }
    )

    # Jeśli mimo wszystko współrzędne nie są floatami, odkomentuj:
    # for col in ['start_latitude','start_longitude','end_latitude','end_longitude','latitude','longitude']:
    #     df[col] = df[col].astype(str).str.replace(',', '.').astype(float)

    df.to_parquet(
        output_parquet,
        engine='pyarrow',
        index=False,
        compression='snappy'
    )

if __name__ == '__main__':
    csv_file = 'kiel_anomalies_fixed.csv'
    parquet_file = 'kiel_anomalies_labeled_2.parquet'
    csv_to_parquet(csv_file, parquet_file)
    print(f'✔ Plik zapisano jako: {parquet_file}')
