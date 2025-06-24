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
# LSTM utils
# --------------------------------------------------------------------------- #

import torch
from LSTM.lstm_encoder import LSTMModel


def create_sequences(data, seq_length):
    """Creates sequences from time-series data for LSTM."""
    xs = []
    for i in range(len(data) - seq_length):
        xs.append(data[i: i + seq_length])
    return np.array(xs)


def get_scores_from_lstm_model(model_artifacts, trip_features):
    """Handle LSTM autoencoder model."""

    # Load model configuration
    model_config = model_artifacts.get("model_config", {})
    input_size = model_config.get("input_size", 5)
    hidden_size = model_config.get("hidden_size", 128)
    num_layers = model_config.get("num_layers", 1)
    sequence_length = model_config.get("sequence_length", 10)
    threshold_percentile = model_config.get("threshold_percentile", 95)

    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LSTMModel(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        output_size=input_size
    )

    # Load state dict
    model.load_state_dict(model_artifacts["model_state"])
    model = model.to(device)
    model.eval()

    print("LSTM model loaded and set to evaluation mode.")

    # Load scaler and prepare data
    scaler = model_artifacts["scaler"]
    scaled_features = scaler.transform(trip_features)

    print("Features scaled using the provided scaler.")

    print("Creating sequences for LSTM...")
    # Create sequences
    if len(scaled_features) < sequence_length:
        # Handle short trips by padding or using available data
        # Note not sure if this is the best way to handle short trips
        sequences = np.array([scaled_features])
        sequences = np.pad(sequences, ((0, 0), (0, max(0, sequence_length - len(scaled_features))), (0, 0)), 'edge')
    else:
        sequences = create_sequences(scaled_features, sequence_length)

    # Get reconstruction errors
    X_tensor = torch.from_numpy(sequences).float().to(device)

    print("Computing reconstruction errors...")
    reconstruction_errors = model.get_reconstruction_error(X_tensor)

    print("Reconstruction errors computed.")
    # Calculate threshold and scores
    threshold = model_artifacts.get("threshold", np.percentile(reconstruction_errors, threshold_percentile))

    # Convert reconstruction errors to anomaly scores for all points
    scores = np.zeros(len(trip_features))

    if len(scaled_features) < sequence_length:
        # For short trips, assign the single reconstruction error to all points
        scores[:] = reconstruction_errors[0]
    else:
        # For normal trips, assign reconstruction errors to corresponding points
        for i, error in enumerate(reconstruction_errors):
            start_idx = i
            end_idx = min(i + sequence_length, len(scores))
            scores[start_idx:end_idx] = np.maximum(scores[start_idx:end_idx], error)

    print("Anomaly scores calculated.")
    return scores, threshold


# --------------------------------------------------------------------------- #
# geometry helpers – identical to training pipeline
# --------------------------------------------------------------------------- #
def haversine(lat1, lon1, lat2, lon2):
    """Vectorized haversine distance (km)."""
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
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

    # Delta features
    trip["dv"] = trip.speed_over_ground.diff().abs().fillna(0)
    dcourse = trip.course_over_ground.diff().abs()
    dcourse = dcourse.where(dcourse <= 180, 360 - dcourse)
    trip["dcourse"] = dcourse.fillna(0)
    trip["ddraft"] = trip.draught.diff().abs().fillna(0)

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


def get_scores_from_model(model, X):
    # Handle sklearn-style models
    if hasattr(model, "predict_proba"):
        # Logistic regression or any classifier with probabilities
        return model.predict_proba(X)[:, 1]  # Class 1 = anomaly
    elif hasattr(model, "decision_function"):
        # OC-SVM, Isolation Forest, etc.
        return -model.decision_function(X)  # negate: lower score = more anomalous
    else:
        raise RuntimeError("Model does not support scoring")


def main() -> None:
    ap = argparse.ArgumentParser("Score single trip and dump JSON")
    ap.add_argument("trip_id")
    ap.add_argument("-i", "--input", default="all_anomalies_combined.parquet")
    # ap.add_argument("-o", "--output", default="/workspace/frontend/src/assets/trip.json")
    ap.add_argument("-o", "--output", default="../frontend/src/assets/trip.json")

    # ap.add_argument("--dispatcher", default="models_per_route/dispatcher.pkl")
    # ap.add_argument("--dispatcher", default="models_per_route_iso_for/dispatcher.pkl")
    # ap.add_argument("--dispatcher", default="models_per_route_lr/dispatcher.pkl")
    ap.add_argument("--dispatcher", default="models_per_route_lstm_ae/dispatcher.pkl")


    ap.add_argument("--model-type", choices=["sklearn", "lstm"], default="lstm",
                    help="Type of model to use")

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

    artifacts = joblib.load(model_path)
    preds = ()

    if args.model_type == "lstm":
        # LSTM current feature columns
        feats = ['latitude', 'longitude',
                    'speed_over_ground', 'course_over_ground'] # NOTE change when retrain
        X_raw = trip[feats].fillna(0).values

        scores, threshold = get_scores_from_lstm_model(artifacts, X_raw)
        print(f"LSTM model loaded for route {route}, threshold: {threshold:.4f}")
        preds = (scores > threshold).astype(int)
    else:
        # Original sklearn logic
        feats: List[str] = artifacts["features"]
        tau: float = artifacts["tau"]
        X_raw = trip[feats].fillna(0).values

        pipe = artifacts.get("pipeline")
        if pipe is not None:
            if hasattr(pipe, "predict_proba"):
                scores = pipe.predict_proba(X_raw)[:, 1]
            else:
                scores = -pipe.decision_function(X_raw)
        else:
            scaler: StandardScaler = artifacts["scaler"]
            model = artifacts["model"]
            Xs = scaler.transform(X_raw)
            scores = get_scores_from_model(model, Xs)

        preds = (scores > tau).astype(int)

    y_true = np.array([1 if x else 0 for x in trip.is_anomaly.values])

    out = []
    for i, row in trip.reset_index(drop=True).iterrows():
        out.append({
            "latitude": float(row.latitude),
            "longitude": float(row.longitude),
            "time_stamp": row.time_stamp.isoformat() if pd.notna(row.time_stamp) else None,
            "score": float(scores[i]),
            "is_anomaly_pred": int(preds[i]),
            # "is_anomaly_pred": int(y_true[i])
        })

    Path(args.output).write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"✓ {len(out)} points scored. JSON saved → {args.output}")


if __name__ == "__main__":
    main()
