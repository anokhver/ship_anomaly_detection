#!/usr/bin/env python3
"""
Visual-evaluation helper
========================
* loads a single trip from the parquet,
* recomputes exactly the same per-point features used at training time
  (Δspeed, Δcourse … local XY, distance to average route),
* scores the points with the correct OC-SVM (chosen by route_id),
* dumps a JSON list that the frontend can render.

The numerical logic is identical to the previous version; only structure and
missing-feature handling were added.
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# --------------------------------------------------------------------------- #
# geometry helpers – identical to training pipeline
# --------------------------------------------------------------------------- #
EARTH_R = 6_371.0  # km


def haversine(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_R * np.arcsin(np.sqrt(a))


def zone_label(row, port_coords, r_port=5.0, r_app=15.0) -> str:
    dmin = min(
        haversine(row.latitude, row.longitude, pc[0], pc[1])
        for pc in port_coords.values()
    )
    if dmin < r_port:
        return "port"
    if dmin < r_app:
        return "approach"
    return "open_sea"


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


# --------------------------------------------------------------------------- #
# feature engineering for ONE trip (mirrors training code)
# --------------------------------------------------------------------------- #
BASE_FEATS = [
    "speed_over_ground", "dv", "dcourse", "ddraft",
    "zone_port", "zone_approach", "zone_open_sea",
    "x_km", "y_km", "dist_to_ref", "route_dummy"
]


def build_feature_frame(trip: pd.DataFrame, full_df: pd.DataFrame) -> pd.DataFrame:
    """Adds all model features to *trip* and returns the enriched frame."""
    route = trip.start_port.iloc[0]

    # Δ features
    trip = trip.sort_values("time_stamp")
    trip["dv"] = trip["speed_over_ground"].diff().abs().fillna(0)
    trip["dcourse"] = trip["course_over_ground"].diff().abs().fillna(0)
    trip["ddraft"] = trip["draught"].diff().abs().fillna(0)

    # zones (needs global port coords)
    port_coords = (
        full_df.groupby("start_port")[["start_latitude", "start_longitude"]]
        .first()
        .apply(tuple, axis=1)
        .to_dict()
    )
    trip["zone"] = trip.apply(zone_label, axis=1, args=(port_coords,))
    for z in ("port", "approach", "open_sea"):
        trip[f"zone_{z}"] = (trip["zone"] == z).astype(int)

    # local projection
    lat0, lon0 = trip.latitude.mean(), trip.longitude.mean()
    kx = 111.320 * np.cos(np.deg2rad(lat0))
    ky = 110.574
    trip["x_km"] = (trip.longitude - lon0) * kx
    trip["y_km"] = (trip.latitude - lat0) * ky

    # distance to average route of THIS route (re-use cache if many trips)
    if "_avg_cache" not in globals():
        globals()["_avg_cache"] = {}
    if route not in globals()["_avg_cache"]:
        globals()["_avg_cache"][route] = average_route(
            full_df[full_df.start_port == route]
        )
    avg = globals()["_avg_cache"][route]
    # map cumulative fraction
    frac = np.zeros(len(trip))
    d = haversine(
        trip.latitude.values[1:], trip.longitude.values[1:],
        trip.latitude.values[:-1], trip.longitude.values[:-1]
    )
    cum = np.concatenate(([0], np.cumsum(d)))
    total = cum[-1] if cum[-1] > 0 else 1.0
    frac[:] = cum / total
    trip["dist_to_ref"] = [
        haversine(lat, lon, avg[int(f * 99), 0], avg[int(f * 99), 1])
        for lat, lon, f in zip(trip.latitude, trip.longitude, frac)
    ]

    trip["route_dummy"] = 1
    return trip


# --------------------------------------------------------------------------- #
# main CLI utility
# --------------------------------------------------------------------------- #
def load_dispatcher(path: str) -> Dict[str, str]:
    try:
        return joblib.load(path)
    except Exception as e:
        sys.exit(f"Cannot load dispatcher @ {path}: {e}")


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
    scaler: StandardScaler = artefacts["scaler"]
    model: OneClassSVM = artefacts["model"]
    tau: float = artefacts["tau"]
    feats: List[str] = artefacts["features"]

    Xs = scaler.transform(trip[feats].fillna(0).values)
    scores = -model.decision_function(Xs)
    preds = (scores > tau).astype(int)

    # JSON output
    out = []
    for i, row in trip.reset_index(drop=True).iterrows():
        out.append(
            {
                "latitude": float(row.latitude),
                "longitude": float(row.longitude),
                "time_stamp": (
                    row.time_stamp.isoformat() if pd.notna(row.time_stamp) else None
                ),
                "score": float(scores[i]),
                "is_anomaly_pred": int(preds[i]),
            }
        )

    Path(args.output).write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"✓ {len(out)} points scored. JSON saved → {args.output}")


if __name__ == "__main__":
    main()
