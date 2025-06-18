#!/usr/bin/env python3
import pandas as pd

# 1. Wczytaj dane
df = pd.read_parquet("all_anomalies_combined.parquet", engine="pyarrow")

# 2. Wyodrębnij pojedynczy trip i posortuj po czasie
mask = df["trip_id"] == 29165
trip = df.loc[mask].sort_values("time_stamp").copy()

# 3. Wyznacz oryginalne indeksy dla punktów 812–1259 (0-based)
#    iloc[812:1260] zwraca wiersze o pozycjach [812, 813, …, 1259]
indices_to_label = trip.iloc[563:1095].index

# 4. Oznacz je jako anomalie
df.loc[indices_to_label, "is_anomaly"] = True
# Jeśli używasz kolumny y_true:
if "y_true" in df.columns:
    df.loc[indices_to_label, "y_true"] = 1

# 5. Zapisz zmodyfikowany DataFrame
df.to_parquet("all_anomalies_combined.parquet", engine="pyarrow", index=False)

print(f"Labeled {len(indices_to_label)} points in trip 240295 as anomalies.")
