import pandas as pd

df = pd.read_parquet("all_anomalies_combined.parquet", engine="pyarrow")

mask = df["trip_id"] == 29165
trip = df.loc[mask].sort_values("time_stamp").copy()

indices_to_label = trip.iloc[563:1095].index

df.loc[indices_to_label, "is_anomaly"] = True
if "y_true" in df.columns:
    df.loc[indices_to_label, "y_true"] = 1

df.to_parquet("all_anomalies_combined.parquet", engine="pyarrow", index=False)

print(f"Labeled {len(indices_to_label)} points in trip 240295 as anomalies.")
