"""
Isolation forest per-route training pipeline (with simple v grid-search + τ thresholding)
-------------------------------------------------------------------------------
• cleans and featurizes AIS data
• trains a separate Isolation Forest for each route
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
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore", category=UserWarning)

# ───────────────────────────── configuration ────────────────────────────── #
DROP_TRIPS      = []
BASE_COLUMNS    = [
    "speed_over_ground", "dv", "dcourse", "ddraft",
    "zone",
    "x_km", "y_km", "dist_to_ref", "route_dummy"
]
ZONES           = [[53.8, 53.5, 8.6, 8.14], [53.66, 53.0, 11.0, 9.5], [54.45, 54.2, 10.3, 10.0], [55.0, 54.25, 18.9, 18.2]]  # [lat_max, lat_min, lon_max, lon_min]
N_ESTIMATORS        = [100, 101]
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
CONTAMINATION_BREMERHAVEN = 0.02
CONTAMINATION_KIEL = 0.008
TEST_FRACTION_N = 0.125
def train_per_route(df: pd.DataFrame,
                    out_dir: str = "models_per_route_iso_for") -> None:
    """Train one Isolation Forest per *route_id* found in *df*.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns ``route_id``, ``y_true`` (0 = normal, 1 = anomaly)
        and all the feature columns listed in ``BASE_COLUMNS``.
    out_dir : str, default="models_per_route_iso_for"
        Where the per‑route models and *dispatcher.pkl* should be written.
    """

    Path(out_dir).mkdir(exist_ok=True)
    dispatcher: Dict[str, str] = {}

    for route in df.route_id.unique():
        t0 = time.time()
        print(f"\n=== Training route: {route} ===")

        # route‑specific feature engineering (user‑defined helper)
        fr = add_route_specific_features(df, route)

        # split route frame into arrays
        X_norm = fr[fr.y_true == 0][BASE_COLUMNS].fillna(0).values
        X_anom = fr[fr.y_true == 1][BASE_COLUMNS].fillna(0).values

        if X_norm.size == 0:
            print("  * No normal points, skipping this route.")
            continue

        # ── build a small, stratified test set ──────────────────────────── #
        idx_anom = fr[fr.y_true == 1].index.to_numpy()
        n_norm_test = max(1, int(TEST_FRACTION_N * len(X_norm)))
        idx_norm = (
            fr[fr.y_true == 0]
            .sample(n=n_norm_test, random_state=42)
            .index.to_numpy()
        )

        X_test = np.vstack([
            fr.loc[idx_anom, BASE_COLUMNS].fillna(0).values,
            fr.loc[idx_norm, BASE_COLUMNS].fillna(0).values,
        ])
        y_test = np.concatenate([
            np.ones(len(idx_anom), dtype=int),
            np.zeros(len(idx_norm), dtype=int),
        ])

        # contamination fraction depends on the port
        contamination = (
            CONTAMINATION_BREMERHAVEN if route == "BREMERHAVEN" else CONTAMINATION_KIEL
        )

        # ── grid‑search over *n_estimators* ─────────────────────────────── #
        best = {"auc": -np.inf}
        for n_est in N_ESTIMATORS:
            pipe = Pipeline(
                steps=[
                    ("scaler", StandardScaler(with_mean=False)),
                    (
                        "iso",
                        IsolationForest(
                            n_estimators=n_est,
                            contamination=contamination,
                            max_samples="auto",
                            random_state=42,
                            n_jobs=-1,
                            verbose=0,
                        ),
                    ),
                ]
            )

            # fit only on known‑normal data (unsupervised)
            pipe.fit(X_norm)

            # IsolationForest.decision_function -> higher is *more normal*
            # We flip the sign so that higher = *more anomalous* like OC‑SVM.
            scores_train = -pipe.decision_function(X_norm)
            # tau = np.percentile(scores_train, 100 * (1 - contamination))
            tau = np.percentile(scores_train, 94)

            scores_test = -pipe.decision_function(X_test)
            preds = (scores_test > tau).astype(int)

            auc = (
                roc_auc_score(y_test, scores_test) if len(np.unique(y_test)) > 1 else 0.0
            )
            print(f"  n_estimators={n_est:<4}  τ={tau:6.3f}  AUC={auc:5.3f}")

            if auc > best["auc"]:
                best.update(pipe=pipe, n_est=n_est, tau=tau, auc=auc)

        # ── final evaluation and serialisation ──────────────────────────── #
        print(
            f"\n-> Selected n_estimators={best['n_est']}  τ={best['tau']:.3f}  AUC={best['auc']:.3f}"
        )
        scores_test = -best["pipe"].decision_function(X_test)
        preds = (scores_test > best["tau"]).astype(int)

        print(confusion_matrix(y_test, preds))
        print(classification_report(y_test, preds, digits=3))
        print(f"Route {route} done in {time.time() - t0:.1f}s\n")

        model_path = Path(out_dir) / f"iso_{route}.pkl"
        joblib.dump(
            {"pipeline": best["pipe"], "features": BASE_COLUMNS, "tau": best["tau"]},
            model_path,
        )
        dispatcher[route] = str(model_path)

    # save dispatcher mapping route → model path
    joblib.dump(dispatcher, Path(out_dir) / "dispatcher.pkl")
    print("All models saved, dispatcher.pkl created.")


if __name__ == "__main__":
    df_all = load_and_prepare("all_anomalies_combined.parquet")
    train_per_route(df_all)
