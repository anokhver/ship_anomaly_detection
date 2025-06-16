"""
OC-SVM per-route training pipeline
---------------------------------
* loads the combined AIS parquet
* cleans / filters data (dates, unwanted trips, missing ship type ...)
* engineers per-point features (∆speed, ∆course, local XY, distance-to-route)
* performs PU self-training and ν / τ search per route
* stores one model file per route + a dispatcher with paths
The numerical logic is unchanged – only the structure, naming and docstrings
were improved for readability and maintenance.
"""
from __future__ import annotations
import os
import time
import warnings
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

warnings.filterwarnings("ignore", category=UserWarning)

# --------------------------------------------------------------------------- #
# geometry helpers
# --------------------------------------------------------------------------- #
EARTH_R = 6_371.0  # km


def haversine(lat1, lon1, lat2, lon2) -> float | np.ndarray:
    """Vectorised great-circle distance (km)."""
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_R * np.arcsin(np.sqrt(a))


# --------------------------------------------------------------------------- #
# data preparation
# --------------------------------------------------------------------------- #
DROP_TRIPS = [10257]                       # trips to exclude completely
N_UL_SAMPLES = 10_000                      # unlabeled cap for PU-learning
BASE_COLUMNS = [
    "speed_over_ground", "dv", "dcourse", "ddraft",
    "zone_port", "zone_approach", "zone_open_sea",
    "x_km", "y_km", "dist_to_ref",          # engineered numerics
    "route_dummy"                           # one-hot for the current route
]


def load_and_clean(path: str) -> pd.DataFrame:
    """Load parquet, drop unwanted trips, parse dates, add y_true + route_id."""
    df = pd.read_parquet(path, engine="pyarrow")
    print(f"Loaded {len(df):,} rows – dropping {len(df[df.trip_id.isin(DROP_TRIPS)]):,}")
    df = df[~df.trip_id.isin(DROP_TRIPS)].reset_index(drop=True)

    for col in ("start_time", "end_time", "time_stamp"):
        df[col] = pd.to_datetime(df[col])

    df = df.dropna(subset=["ship_type"]).reset_index(drop=True)
    df["y_true"] = df["is_anomaly"].map({True: 1, False: 0})
    df["route_id"] = df["start_port"]
    return df


def add_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    """∆speed/∆course/∆draft + categorical zone one-hots."""
    port_coords = (
        df.groupby("start_port")[["start_latitude", "start_longitude"]]
        .first().to_dict("index")
    )
    R_PORT, R_APP = 5.0, 15.0

    def zone_label(row) -> str:
        dmin = min(
            haversine(row.latitude, row.longitude,
                      p["start_latitude"], p["start_longitude"])
            for p in port_coords.values()
        )
        if dmin < R_PORT:
            return "port"
        if dmin < R_APP:
            return "approach"
        return "open_sea"

    df = df.sort_values(["trip_id", "time_stamp"])
    df["dv"] = df.groupby("trip_id")["speed_over_ground"].diff().abs().fillna(0)
    df["dcourse"] = df.groupby("trip_id")["course_over_ground"].diff().abs().fillna(0)
    df["ddraft"] = df.groupby("trip_id")["draught"].diff().abs().fillna(0)
    df["zone"] = df.apply(zone_label, axis=1)
    df = pd.concat([df, pd.get_dummies(df["zone"], prefix="zone")], axis=1)
    return df


# --------------------------------------------------------------------------- #
# route-specific helpers
# --------------------------------------------------------------------------- #
def compute_average_route(df_route: pd.DataFrame, n_points: int = 100) -> np.ndarray:
    """
    Average trajectory for a route (shape = (n_points, 2)).
    Each trip is resampled to *n_points* by cumulative path fraction,
    then averaged.
    """
    segments = []
    for _, trip in df_route.groupby("trip_id"):
        trip = trip.sort_values("time_stamp")
        lat, lon = trip.latitude.to_numpy(), trip.longitude.to_numpy()
        d = haversine(lat[1:], lon[1:], lat[:-1], lon[:-1])
        cum = np.concatenate(([0], np.cumsum(d)))
        total = cum[-1]
        if total <= 0:
            continue
        frac = cum / total
        target = np.linspace(0, 1, n_points)
        segments.append(np.vstack([
            np.interp(target, frac, lat),
            np.interp(target, frac, lon)
        ]).T)
    return np.mean(np.stack(segments, axis=0), axis=0)


def prepare_route_frame(df_full: pd.DataFrame, route: str) -> pd.DataFrame:
    """Adds local XY projection and distance-to-average for a single route."""
    df_r = df_full[df_full.route_id == route].copy()

    # local meter projection
    lat0, lon0 = df_r.latitude.mean(), df_r.longitude.mean()
    kx = 111.320 * np.cos(np.deg2rad(lat0))
    ky = 110.574
    df_r["x_km"] = (df_r.longitude - lon0) * kx
    df_r["y_km"] = (df_r.latitude - lat0) * ky

    # average trajectory & per-point distance
    avg = compute_average_route(df_r)
    idx_map = df_r.index
    frac = np.zeros(len(df_r))
    for _, trip in df_r.groupby("trip_id"):
        pos = idx_map.get_indexer(trip.index)
        lat, lon = trip.latitude.values, trip.longitude.values
        d = haversine(lat[1:], lon[1:], lat[:-1], lon[:-1])
        cum = np.concatenate(([0], np.cumsum(d)))
        total = cum[-1] if cum[-1] > 0 else 1.0
        frac[pos] = cum / total
    df_r["dist_to_ref"] = [
        haversine(lat, lon, avg[int(f * 99), 0], avg[int(f * 99), 1])
        for lat, lon, f in zip(df_r.latitude, df_r.longitude, frac)
    ]

    df_r["route_dummy"] = 1.0  # constant feature for this route
    return df_r


# --------------------------------------------------------------------------- #
# model training
# --------------------------------------------------------------------------- #
def train_route_models(df: pd.DataFrame,
                       out_dir: str = "models_per_route",
                       max_unlabeled: int = N_UL_SAMPLES) -> None:
    """Trains one OC-SVM per route and stores a dispatcher with file paths."""
    os.makedirs(out_dir, exist_ok=True)
    dispatcher: Dict[str, str] = {}

    for route in df.route_id.unique():
        t0 = time.time()
        print(f"\n=== Route {route} ===")
        fr = prepare_route_frame(df, route)

        fr.loc[fr["zone_port"] == 1, "y_true"] = 0  # port zone is normal

        X = fr[BASE_COLUMNS].fillna(0).values
        y = fr["y_true"].values
        X_norm, X_anom = X[y == 0], X[y == 1]
        X_unl = X[np.isnan(y)]
        if len(X_unl) > max_unlabeled:
            X_unl = X_unl[np.random.choice(len(X_unl), max_unlabeled, replace=False)]

        # ---- PU self-training ------------------------------------------------
        scaler = StandardScaler().fit(X_norm)
        model = OneClassSVM(kernel="rbf", nu=0.05, gamma="scale").fit(scaler.transform(X_norm))
        for _ in range(2):
            scores_u = -model.decision_function(scaler.transform(X_unl))
            keep = scores_u > np.percentile(scores_u, 90)
            X_pseudo = X_unl[~keep]
            if not len(X_pseudo):
                break
            X_norm = np.vstack([X_norm, X_pseudo])
            X_unl = X_unl[keep]
            scaler = StandardScaler().fit(X_norm)
            model = OneClassSVM(kernel="rbf", nu=0.05, gamma="scale").fit(scaler.transform(X_norm))

        # ---- ν grid + dynamic τ --------------------------------------------
        best = {"auc": -np.inf}
        sample_norm = X_norm[np.random.choice(len(X_norm), int(0.1 * len(X_norm)), replace=False)]
        X_test = np.vstack([X_anom, sample_norm])
        y_test = np.concatenate([np.ones(len(X_anom)), np.zeros(len(sample_norm))])

        for nu in (0.01, 0.2):
            clf = OneClassSVM(kernel="rbf", nu=nu, gamma="scale").fit(scaler.transform(X_norm))
            tau = np.percentile(-clf.decision_function(scaler.transform(X_norm)), 100 * (1 - nu))
            scores = -clf.decision_function(scaler.transform(X_test))
            auc = roc_auc_score(y_test, scores)
            print(f"  ν={nu:<4} τ={tau:.3f} AUC={auc:.3f}")
            if auc > best["auc"]:
                best.update(model=clf, tau=tau, auc=auc)

        # ---- evaluation ------------------------------------------------------
        scores = -best["model"].decision_function(scaler.transform(X_test))
        preds = (scores > best["tau"]).astype(int)
        print(confusion_matrix(y_test, preds))
        print(classification_report(y_test, preds, digits=3))
        print(f"ROC AUC: {best['auc']:.3f}  |  time: {time.time() - t0:.1f}s")

        # ---- save ------------------------------------------------------------
        path = os.path.join(out_dir, f"ocsvm_{route}.pkl")
        joblib.dump({"model": best["model"], "scaler": scaler,
                     "features": BASE_COLUMNS, "tau": best["tau"]},
                    path)
        dispatcher[route] = path

    joblib.dump(dispatcher, os.path.join(out_dir, "dispatcher.pkl"))
    print("\nAll route models saved.")


# --------------------------------------------------------------------------- #
# entry-point
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    df_raw = load_and_clean("all_anomalies_combined.parquet")
    df_feat = add_basic_features(df_raw)
    train_route_models(df_feat)