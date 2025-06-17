#!/usr/bin/env python3
"""
Visual-evaluation helper
========================
* loads a single trip from the parquet,
* recomputes exactly the same per-point features used at training time
  (Δspeed, Δcourse … local XY, distance to average route),
* scores the points with the correct OC-SVM (chosen by route_id),
* dumps a JSON list that the frontend can render.

Usage and output remain unchanged.
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# --------------------------------------------------------------------------- #
# configuration – same as training
# --------------------------------------------------------------------------- #
ZONES = [[53.8, 53.5, 8.6, 8.14], [53.66, 53.0, 11.0, 9.5]]  # [lat_max, lat_min, lon_max, lon_min]
EARTH_R = 6_371.0  # km

# --------------------------------------------------------------------------- #
# geometry helpers – identical to training pipeline
# --------------------------------------------------------------------------- #
def haversine(lat1, lon1, lat2, lon2):
    """Vectorized haversine distance (km)."""
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    return 2 * EARTH_R * np.arcsin(np.sqrt(a))


def zone_label(row) -> int:
    """Assign zone exactly as in training: 0 if within any defined rectangle, else 1."""
    for lat_max, lat_min, lon_max, lon_min in ZONES:
        if lat_min <= row.latitude <= lat_max and lon_min <= row.longitude <= lon_max:
            return 0
    return 1


def load_dispatcher(path: str) -> Dict[str, str]:
    """Load the dispatcher.pkl or exit with an error."""
    try:
        return joblib.load(path)
    except Exception as e:
        sys.exit(f"Cannot load dispatcher at {path}: {e}")


def average_route(df_route: pd.DataFrame, n_points: int = 100) -> np.ndarray:
    """Weighted average trajectory, same routine as in training."""
    segments: List[np.ndarray] = []
    for _, trip in df_route.groupby("trip_id"):
        trip = trip.sort_values("time_stamp")
        lat, lon = trip.latitude.values, trip.longitude.values
        d = haversine(lat[1:], lon[1:], lat[:-1], lon[:-1])
        cum = np.concatenate(([0], np.cumsum(d)))
        if cum[-1] <= 0:
            continue
        frac = cum / cum[-1]
        tgt = np.linspace(0, 1, n_points)
        segments.append(
            np.vstack([np.interp(tgt, frac, lat), np.interp(tgt, frac, lon)]).T
        )
    return np.mean(np.stack(segments, axis=0), axis=0)  # (n_points, 2)


def build_feature_frame(trip: pd.DataFrame, full_df: pd.DataFrame) -> pd.DataFrame:
    """Adds all model features to *trip* and returns the enriched frame, matching training pipeline."""
    route = trip.start_port.iloc[0]

    # Δ features
    trip = trip.sort_values("time_stamp").copy()
    trip["dv"] = trip["speed_over_ground"].diff().abs().fillna(0)
    trip["dcourse"] = trip["course_over_ground"].diff().abs().fillna(0)
    trip["ddraft"] = trip["draught"].diff().abs().fillna(0)

    # zone
    trip["zone"] = trip.apply(zone_label, axis=1)

    # local projection: use full route mean coordinates as in training
    df_route = full_df[full_df.start_port == route]
    lat0, lon0 = df_route.latitude.mean(), df_route.longitude.mean()
    kx = 111.320 * np.cos(np.deg2rad(lat0))
    ky = 110.574
    trip["x_km"] = (trip.longitude - lon0) * kx
    trip["y_km"] = (trip.latitude - lat0) * ky

    # distance to average trajectory
    if "_avg_cache" not in globals():
        globals()["_avg_cache"] = {}
    if route not in globals()["_avg_cache"]:
        globals()["_avg_cache"][route] = average_route(df_route)
    avg = globals()["_avg_cache"][route]

    # cumulative fraction along trip
    d = haversine(
        trip.latitude.values[1:], trip.longitude.values[1:],
        trip.latitude.values[:-1], trip.longitude.values[:-1]
    )
    cum = np.concatenate(([0], np.cumsum(d)))
    total = cum[-1] if cum[-1] > 0 else 1.0
    frac = cum / total
    trip["dist_to_ref"] = [
        haversine(lat, lon, avg[int(f * (len(avg) - 1)), 0], avg[int(f * (len(avg) - 1)), 1])
        for lat, lon, f in zip(trip.latitude, trip.longitude, frac)
    ]

    # constant dummy for route
    trip["route_dummy"] = 1.0

    return trip


def main() -> None:
    ap = argparse.ArgumentParser("Score single trip and dump JSON")
    ap.add_argument("trip_id")
    ap.add_argument("-i", "--input", default="all_anomalies_combined.parquet")
    ap.add_argument("-o", "--output", default="/workspace/frontend/src/assets/trip.json")
    ap.add_argument("--dispatcher", default="models_per_route/dispatcher.pkl")
    args = ap.parse_args()

    dispatcher = load_dispatcher(args.dispatcher)

    df_all = pd.read_parquet(args.input, engine="pyarrow")
    tid = int(args.trip_id) if df_all.trip_id.dtype.kind in "iu" else args.trip_id
    trip = df_all[df_all.trip_id == tid].copy()
    if trip.empty:
        sys.exit(f"[!] Trip {tid} not found in {args.input}")

    trip = build_feature_frame(trip, df_all)

    route = trip.start_port.iloc[0]
    model_path = dispatcher.get(route)
    if not model_path:
        sys.exit(f"[!] No model for route {route}")

    artefacts = joblib.load(model_path)
    feats: List[str] = artefacts["features"]
    tau: float = artefacts["tau"]

    X_raw = trip[feats].fillna(0).values

    pipe = artefacts.get("pipeline")
    if pipe is not None:
        scores = -pipe.decision_function(X_raw)
    else:
        scaler: StandardScaler = artefacts["scaler"]
        model = artefacts["model"]
        Xs = scaler.transform(X_raw)
        scores = -model.decision_function(Xs)

    preds = (scores > tau).astype(int)

    out = []
    for i, row in trip.reset_index(drop=True).iterrows():
        out.append({
            "latitude": float(row.latitude),
            "longitude": float(row.longitude),
            "time_stamp": row.time_stamp.isoformat() if pd.notna(row.time_stamp) else None,
            "score": float(scores[i]),
            "is_anomaly_pred": int(preds[i]),
        })

    Path(args.output).write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"✓ {len(out)} points scored. JSON saved → {args.output}")


if __name__ == "__main__":
    main()
