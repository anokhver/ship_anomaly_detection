"""
Data Preparation Pipeline for Vessel Trajectory Anomaly Detection

This module preprocesses AIS vessel trajectory data for LSTM-based anomaly detection.
It adds engineered features like delta values, zone classifications, and route-specific
deviation metrics that help identify abnormal vessel behaviors.
"""

import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path

# Enable progress bars for pandas operations
tqdm.pandas()

# Configuration constants
DROP_TRIPS = [10257]  # Problematic trips to exclude
ZONES = [[53.8, 53.5, 8.6, 8.14], [53.66, 53.0, 11.0, 9.5]]  # Port zone boundaries
R_PORT, R_APP = 5.0, 15.0  # Port and approach radii in km
EARTH_R = 6_371.0  # Earth radius in km
RANDOM_STATE = 42


def haversine(lat1, lon1, lat2, lon2):
    """Calculate great-circle distance between points in kilometers."""
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))

    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_R * np.arcsin(np.sqrt(a))


def load_and_prepare(path: str) -> pd.DataFrame:
    """
    Load vessel data and add basic features.

    Features added:
    - Delta features (dv, dcourse, ddraft) for detecting sudden changes
    - Zone classification (port area vs open water)
    - Binary anomaly labels for supervised learning
    """
    # Load data and remove problematic trips
    df = pd.read_parquet(path, engine="pyarrow")
    print(f"Loaded {len(df):,} rows, dropping {len(df[df.trip_id.isin(DROP_TRIPS)]):,} rows from {DROP_TRIPS}")
    df = df[~df.trip_id.isin(DROP_TRIPS)].reset_index(drop=True)

    # Parse datetime columns
    for col in ("start_time", "end_time", "time_stamp"):
        df[col] = pd.to_datetime(df[col])

    # Create binary labels and route identifier
    df["y_true"] = df["is_anomaly"].map({True: 1, False: 0})
    df["route_id"] = df["start_port"]

    # Compute delta features (changes between consecutive points)
    df = df.sort_values(["trip_id", "time_stamp"])
    df["dv"] = df.groupby("trip_id")["speed_over_ground"].diff().abs().fillna(0)
    df["dcourse"] = df.groupby("trip_id")["course_over_ground"].diff().abs().fillna(0)
    df["ddraft"] = df.groupby("trip_id")["draught"].diff().abs().fillna(0)

    # Zone classification (0=port area, 1=open water)
    def zone_label(row) -> int:
        """Check if vessel is in port zone."""
        for lat_max, lat_min, lon_max, lon_min in ZONES:
            if lat_min <= row.latitude <= lat_max and lon_min <= row.longitude <= lon_max:
                return 0
        return 1

    df["zone"] = df.progress_apply(zone_label, axis=1)
    df = pd.concat([df, pd.get_dummies(df["zone"], prefix="zone")], axis=1)

    return df


def compute_average_route(df_route: pd.DataFrame, n_points: int = 100) -> np.ndarray:
    """
    Compute reference trajectory by averaging all trips on a route.

    Returns array of shape (n_points, 2) with [lat, lon] coordinates.
    """
    segments = []

    for _, trip in df_route.groupby("trip_id"):
        trip = trip.sort_values("time_stamp")
        lat, lon = trip.latitude.to_numpy(), trip.longitude.to_numpy()

        # Calculate cumulative distance along trip
        d = haversine(lat[1:], lon[1:], lat[:-1], lon[:-1])
        cum = np.concatenate(([0], np.cumsum(d)))

        if cum[-1] <= 0:  # Skip trips with no movement
            continue

        # Normalize to fraction of total distance
        frac = cum / cum[-1]

        # Resample to fixed number of points
        target = np.linspace(0, 1, n_points)
        segments.append(np.vstack([
            np.interp(target, frac, lat),
            np.interp(target, frac, lon)
        ]).T)

    if not segments:
        return np.array([])

    # Average corresponding points across all trips
    return np.mean(np.stack(segments, axis=0), axis=0)


def add_route_specific_features(df: pd.DataFrame, route: str) -> pd.DataFrame:
    """
    Add features specific to each route:
    - Local coordinates (x_km, y_km) for distance calculations
    - Distance to reference trajectory (dist_to_ref)
    """
    df_r = df[df.route_id == route].copy()

    # Create local coordinate system centered on route
    lat0, lon0 = df_r.latitude.mean(), df_r.longitude.mean()
    kx = 111.320 * np.cos(np.deg2rad(lat0))  # Longitude to km
    ky = 110.574  # Latitude to km
    df_r["x_km"] = (df_r.longitude - lon0) * kx
    df_r["y_km"] = (df_r.latitude - lat0) * ky

    # Compute reference trajectory
    avg = compute_average_route(df_r)
    if avg.size == 0:
        df_r["dist_to_ref"] = 0.0
        df_r["route_dummy"] = 1.0
        return df_r

    # Calculate distance to reference trajectory
    idx_map = df_r.index
    frac = np.zeros(len(df_r))

    for _, trip in tqdm(df_r.groupby("trip_id"), desc=f"Processing trips for route {route}"):
        pos = idx_map.get_indexer(trip.index)
        lat, lon = trip.latitude.values, trip.longitude.values
        d = haversine(lat[1:], lon[1:], lat[:-1], lon[:-1])

        cum = np.concatenate(([0], np.cumsum(d)))
        total = cum[-1] if cum[-1] > 0 else 1
        frac[pos] = cum / total

    # Compute distance from each point to reference
    df_r["dist_to_ref"] = [
        haversine(lat, lon, avg[int(f * 99), 0], avg[int(f * 99), 1])
        for lat, lon, f in zip(df_r.latitude, df_r.longitude, frac)
    ]
    df_r["route_dummy"] = 1.0

    return df_r


def preprocess_data(data_path: str, output_dir: str, output_name: str = "LSTM_preprocessed") -> pd.DataFrame:
    """
    Main preprocessing pipeline.

    Args:
        data_path: Path to input parquet file
        output_dir: Directory to save processed data
        output_name: Name for output file (without extension)

    Returns:
        Preprocessed DataFrame ready for LSTM training
    """
    Path(output_dir).mkdir(exist_ok=True)

    # Load and prepare basic features
    df = load_and_prepare(data_path)

    # Process each route separately
    dfs = []
    for route in df.route_id.unique():
        fr = add_route_specific_features(df, route)
        dfs.append(fr)

    # Combine all routes
    df_final = pd.concat(dfs, ignore_index=True)
    df_final.sort_values(["trip_id", "time_stamp"], inplace=True)

    # Save processed data
    output_path = f"{output_dir}/{output_name}.parquet"
    df_final.to_parquet(output_path, index=False)
    print(f"Saved processed data to {output_path}")

    return df_final


if __name__ == "__main__":
    # Example usage
    data_path = "../all_anomalies_combined.parquet"
    output_dir = "data"

    df_final = preprocess_data(data_path, output_dir)
    print(f"Preprocessing complete. Shape: {df_final.shape}")