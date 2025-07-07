"""
OC-SVM per-route training pipeline (with simple v grid-search + τ thresholding)
-------------------------------------------------------------------------------
• cleans and featurizes AIS data
• trains a separate One-Class SVM for each route
• selects best v by ROC-AUC (anomaly vs random normal)
• saves {pipeline, features, τ} + dispatcher.pkl
"""

from __future__ import annotations
import os
import time
import warnings
from pathlib import Path
from typing import Dict

import joblib
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, f1_score, recall_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore", category=UserWarning)

# ───────────────────────────── configuration ────────────────────────────── #
DROP_TRIPS      = []
BASE_COLUMNS    = [
    "speed_over_ground", "dv", "dcourse", "ddraft",
    "zone",
    "x_km", "y_km", "dist_to_ref", "route_dummy"
]
ZONES           = [[53.8, 53.5, 8.6, 8.14], [53.66, 53.0, 11.0, 9.5], [54.45, 54.2, 10.3, 10.0], [54.71, 54.25, 19, 18.35]]  # [lat_max, lat_min, lon_max, lon_min]
TEST_FRACTION_N = 0.10        # fraction of normal points to include in test
R_PORT, R_APP   = 5.0, 15.0   # km: defines "port" and "approach" zones
EARTH_R         = 6_371.0     # Earth radius in km


# ───────────────────────────── helpers ─────────────────────────────────── #
def haversine(lat1, lon1, lat2, lon2):
    """Vectorized haversine distance (km)."""
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_R * np.arcsin(np.sqrt(a))


def load_and_prepare(path: str) -> pd.DataFrame:
    """
    Load parquet, drop specified trips, parse dates, compute delta-features,
    map labels and one-hot port zones.
    """
    df = pd.read_parquet(path, engine="pyarrow")
    print(f"Loaded {len(df):,} rows, dropping {len(df[df.trip_id.isin(DROP_TRIPS)]):,} rows from {DROP_TRIPS}")
    df = df[~df.trip_id.isin(DROP_TRIPS)].reset_index(drop=True)

    for col in ("start_time", "end_time", "time_stamp"):
        df[col] = pd.to_datetime(df[col])

    df = df.dropna(subset=["ship_type"]).reset_index(drop=True)
    df["y_true"]   = df["is_anomaly"].map({True: 1, False: 0})
    df["route_id"] = df["start_port"]

    # per-point deltas
    df = df.sort_values(["trip_id", "time_stamp"])
    df["dv"]      = df.groupby("trip_id")["speed_over_ground"].diff().abs().fillna(0)
    dcourse = df.groupby("trip_id")["course_over_ground"].diff().abs()
    dcourse = dcourse.where(dcourse <= 180, 360 - dcourse)
    df["dcourse"] = dcourse.fillna(0)
    df["ddraft"]  = df.groupby("trip_id")["draught"].diff().abs().fillna(0)

    # zones
    port_coords = (
        df.groupby("start_port")[["start_latitude", "start_longitude"]]
          .first()
          .to_dict("index")
    )

    def _in_any_rect(lat: float, lon: float) -> bool:
        for lat_max, lat_min, lon_max, lon_min in ZONES:
            if lat_min <= lat <= lat_max and lon_min <= lon <= lon_max:
                return True
        return False

    def zone_label(row) -> str:
        if _in_any_rect(row.latitude, row.longitude):
            return 0
        return 1

    df["zone"] = df.apply(zone_label, axis=1)
    df = pd.concat([df, pd.get_dummies(df["zone"], prefix="zone")], axis=1)
    return df


def compute_average_route(df_route: pd.DataFrame, n_points: int = 100) -> np.ndarray:
    """
    Compute average trajectory for a route by resampling each trip to n_points
    along cumulative distance fraction, then averaging.
    """
    segments = []
    for _, trip in df_route.groupby("trip_id"):
        trip = trip.sort_values("time_stamp")
        lat, lon = trip.latitude.to_numpy(), trip.longitude.to_numpy()
        d = haversine(lat[1:], lon[1:], lat[:-1], lon[:-1])
        cum = np.concatenate(([0], np.cumsum(d)))
        if cum[-1] <= 0:
            continue
        frac   = cum / cum[-1]
        target = np.linspace(0, 1, n_points)
        segments.append(np.vstack([np.interp(target, frac, lat),
                                   np.interp(target, frac, lon)]).T)
    return np.mean(np.stack(segments, axis=0), axis=0)


def add_route_specific_features(df: pd.DataFrame, route: str) -> pd.DataFrame:
    """
    For a single route:
    • project lat/lon to local x_km, y_km
    • compute distance to average route (dist_to_ref)
    • add constant route_dummy = 1
    """
    df_r = df[df.route_id == route].copy()

    # local projection
    lat0, lon0 = df_r.latitude.mean(), df_r.longitude.mean()
    kx = 111.320 * np.cos(np.deg2rad(lat0))
    ky = 110.574
    df_r["x_km"] = (df_r.longitude - lon0) * kx
    df_r["y_km"] = (df_r.latitude  - lat0) * ky

    # distance to average trajectory
    avg = compute_average_route(df_r)
    idx_map = df_r.index
    frac = np.zeros(len(df_r))
    for _, trip in df_r.groupby("trip_id"):
        pos = idx_map.get_indexer(trip.index)
        lat, lon = trip.latitude.values, trip.longitude.values
        d = haversine(lat[1:], lon[1:], lat[:-1], lon[:-1])
        cum = np.concatenate(([0], np.cumsum(d)))
        total = cum[-1] if cum[-1] > 0 else 1
        frac[pos] = cum / total

    df_r["dist_to_ref"] = [
        haversine(lat, lon, avg[int(f * 99), 0], avg[int(f * 99), 1])
        for lat, lon, f in zip(df_r.latitude, df_r.longitude, frac)
    ]
    df_r["route_dummy"] = 1.0
    return df_r


# ───────────────────────────── training ────────────────────────────────── #
def train_per_route(df: pd.DataFrame, out_dir: str = "models_per_route") -> None:
    Path(out_dir).mkdir(exist_ok=True)
    dispatcher: Dict[str, str] = {}

    for route in df.route_id.unique():
        t0 = time.time()
        print(f"\n=== Training route: {route} ===")

        fr     = add_route_specific_features(df, route)
        X_norm = fr[fr.y_true == 0][BASE_COLUMNS].fillna(0).values
        X_anom = fr[fr.y_true == 1][BASE_COLUMNS].fillna(0).values

        # Total samples
        print(f"  * Normal points: {len(X_norm)}")
        print(f"  * Anomalous points: {len(X_anom)}")
        if len(X_norm) == 0:
            print("  * No normal points, skipping this route.")
            continue

        # ─── prepare test set: all anomalies + fraction of normals ───
        idx_anom    = fr[fr.y_true == 1].index.to_numpy()
        n_norm_test = max(1, int(TEST_FRACTION_N * len(X_norm)))
        idx_norm = fr[fr.y_true == 0].sample(n=n_norm_test, random_state=42).index.to_numpy()
        X_norm = fr[(fr.y_true == 0) & (~fr.index.isin(idx_norm))][BASE_COLUMNS].fillna(0).values

        X_test = np.vstack([
            fr.loc[idx_anom, BASE_COLUMNS].fillna(0).values,
            fr.loc[idx_norm, BASE_COLUMNS].fillna(0).values
        ])
        y_test = np.concatenate([
            np.ones(len(idx_anom), dtype=int),
            np.zeros(len(idx_norm), dtype=int)
        ])

        # mask for port-zone override
        zone = np.concatenate([
            fr.loc[idx_anom, "zone"].to_numpy(),
            fr.loc[idx_norm, "zone"].to_numpy()
        ]).astype(bool)

        # ─── grid-search ───
        NU_GRID = [0.003]
        GAMMA_GRID = ["scale", "auto", 0.001, 0.01, 0.1, 1.0]
        best = {"f1": -np.inf}  # Compare by F1 score
        for nu in NU_GRID:
            for gamma in GAMMA_GRID:
                pipe = Pipeline([
                    ("scaler", StandardScaler()),
                    ("ocsvm",  OneClassSVM(kernel="rbf", gamma=gamma, nu=nu))
                ])

                pipe.fit(X_norm)
                scores_train = -pipe.decision_function(X_norm)
                scores_test = -pipe.decision_function(X_test)
                
                tau_candidates = np.percentile(scores_train, np.linspace(60, 99.9, 80))

                for tau_candidate in tau_candidates:
                    preds = (scores_test > tau_candidate).astype(int)

                    f1 = f1_score(y_test, preds)
                    auc = roc_auc_score(y_test, scores_test) if len(np.unique(y_test)) > 1 else 0.0

                    if f1 > best.get("f1", -1):  # Compare by F1 score
                        best.update(
                            pipe=pipe,
                            nu=nu,
                            gamma=gamma,
                            tau=tau_candidate,
                            auc=auc,
                            f1=f1
                        )
                tau = best["tau"]
                scores_test = -best["pipe"].decision_function(X_test)

                print(f"Training with ν={nu}  γ={gamma}  τ={tau}")
                preds = (scores_test > tau).astype(int)
                auc = roc_auc_score(y_test, scores_test) if len(np.unique(y_test)) > 1 else 0.0
                f1 = f1_score(y_test, preds)
                print(f"  ν={nu:<4}  γ={gamma}  τ={tau:6.3f}  F1={f1:5.3f}  AUC={auc:5.3f}")

        # ─── final evaluation & save  ───
        print(f"\n-> Selected ν={best['nu']}  τ={best['tau']:.3f}  F1={best['f1']:.3f}")
        scores_test = -best["pipe"].decision_function(X_test)
        preds = (scores_test > best["tau"]).astype(int)
       

        print(confusion_matrix(y_test, preds))
        tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()
        print(f"Total samples: {len(y_test)} TP={tp}, FP={fp}, TN={tn}, FN={fn}")
        print(classification_report(y_test, preds, digits=3))
        print(f"Route {route} done in {time.time() - t0:.1f}s\n")

         # Display feature importances
        perm_importance = permutation_importance(
            best["pipe"], X_test, y_test, scoring=anomaly_f1, random_state=42
        )
        for i, importance in enumerate(perm_importance.importances_mean):
            print(f"Feature {BASE_COLUMNS[i]}: {importance}")

        ##########

        model_path = Path(out_dir) / f"ocsvm_{route}.pkl"
        joblib.dump(
            {
                "pipeline": best["pipe"],
                "features": BASE_COLUMNS,
                "tau": best["tau"],
            },
            model_path,
        )
        dispatcher[route] = str(model_path)

    # save dispatcher
    joblib.dump(dispatcher, Path(out_dir) / "dispatcher.pkl")
    print("All models saved, dispatcher.pkl created.")



# Custom scoring function for anomaly detection (F1 score)
def anomaly_f1(estimator, X, y):
    scores = -estimator.decision_function(X)  # higher = more anomalous
    # Use median as threshold for binary prediction
    threshold = np.median(scores)
    preds = (scores > threshold).astype(int)
    return f1_score(y, preds)



if __name__ == "__main__":
    df_all = load_and_prepare("all_anomalies_combined.parquet")
    train_per_route(df_all)
