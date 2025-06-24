#!/usr/bin/env python3
"""
Visual Evaluation Helper
========================

- Data processing (feature engineering) is encapsulated in DataProcessor
- Model loading and scoring are encapsulated in Scorer
- CLI provides a numeric menu to select dispatcher (1=OC-SVM, 2=Isolation Forest,
  3=Logistic Regression, 4=Random Forest)
- Logging is used for informational and error messages
"""
import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
ZONES: List[List[float]] = [
    [53.8, 53.5, 8.6, 8.14],
    [53.66, 53.0, 11.0, 9.5],
    [54.45, 54.2, 10.3, 10.0],
    [55.0, 54.25, 18.9, 18.2],
]
EARTH_RADIUS_KM: float = 6371.0

# Dispatcher mapping: numeric choice -> dispatcher path
DISPATCHER_PATHS: Dict[int, str] = {
    1: "models_per_route/dispatcher.pkl",           # OC-SVM
    2: "models_per_route_iso_for/dispatcher.pkl",  # Isolation Forest
    3: "models_per_route_lr/dispatcher.pkl",       # Logistic Regression
    4: "models_per_route_rf/dispatcher.pkl",       # Random Forest
}

# --------------------------------------------------------------------------- #
# Geometry and Zone Helpers
# --------------------------------------------------------------------------- #

def haversine(lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    """Compute the great-circle distance between two points (km)"""
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


def zone_label(latitude: float, longitude: float) -> int:
    """Assign zone label: 0 inside any zone rectangle, else 1"""
    for lat_max, lat_min, lon_max, lon_min in ZONES:
        if lat_min <= latitude <= lat_max and lon_min <= longitude <= lon_max:
            return 0
    return 1

# --------------------------------------------------------------------------- #
# Data Processing
# --------------------------------------------------------------------------- #
class DataProcessor:
    def __init__(self, full_df: pd.DataFrame):
        self.full_df = full_df
        self._avg_cache: Dict[Any, np.ndarray] = {}

    def get_trip(self, trip_id: Union[int, str]) -> pd.DataFrame:
        df = self.full_df
        if df.trip_id.dtype.kind in "iu":
            trip_id = int(trip_id)
        trip = df[df.trip_id == trip_id].copy()
        if trip.empty:
            raise ValueError(f"Trip '{trip_id}' not found in dataset")
        return trip

    def average_route(self, route: Any, n_points: int = 100) -> np.ndarray:
        if route in self._avg_cache:
            return self._avg_cache[route]
        df_route = self.full_df[self.full_df.start_port == route]
        segments: List[np.ndarray] = []
        for _, trip in df_route.groupby("trip_id"):
            trip = trip.sort_values("time_stamp")
            lats, lons = trip.latitude.values, trip.longitude.values
            d = haversine(lats[1:], lons[1:], lats[:-1], lons[:-1])
            cum = np.concatenate(([0.0], np.cumsum(d)))
            if cum[-1] <= 0:
                continue
            frac = cum / cum[-1]
            tgt = np.linspace(0, 1, n_points)
            seg = np.vstack([np.interp(tgt, frac, lats), np.interp(tgt, frac, lons)]).T
            segments.append(seg)
        if not segments:
            raise ValueError(f"No valid segments for route {route}")
        avg_route = np.mean(np.stack(segments, axis=0), axis=0)
        self._avg_cache[route] = avg_route
        return avg_route

    def build_feature_frame(self, trip: pd.DataFrame) -> pd.DataFrame:
        trip = trip.sort_values("time_stamp").copy()
        route = trip.start_port.iloc[0]
        # Delta features
        trip["dv"] = trip.speed_over_ground.diff().abs().fillna(0)
        dcourse = trip.course_over_ground.diff().abs()
        dcourse = dcourse.where(dcourse <= 180, 360 - dcourse)
        trip["dcourse"] = dcourse.fillna(0)
        trip["ddraft"] = trip.draught.diff().abs().fillna(0)
        # Zone
        trip["zone"] = trip.apply(lambda r: zone_label(r.latitude, r.longitude), axis=1)
        # Local projection
        df_route = self.full_df[self.full_df.start_port == route]
        lat0, lon0 = df_route.latitude.mean(), df_route.longitude.mean()
        kx, ky = 111.320 * np.cos(np.deg2rad(lat0)), 110.574
        trip["x_km"] = (trip.longitude - lon0) * kx
        trip["y_km"] = (trip.latitude - lat0) * ky
        # Distances to average route
        avg = self.average_route(route)
        d = haversine(trip.latitude.values[1:], trip.longitude.values[1:],
                      trip.latitude.values[:-1], trip.longitude.values[:-1])
        cum = np.concatenate(([0.0], np.cumsum(d)))
        total = cum[-1] if cum[-1] > 0 else 1.0
        frac = cum / total
        distances = [
            haversine(np.array([lat]), np.array([lon]),
                      np.array([avg[int(f*(len(avg)-1)),0]]),
                      np.array([avg[int(f*(len(avg)-1)),1]]))[0]
            for lat, lon, f in zip(trip.latitude, trip.longitude, frac)
        ]
        trip["dist_to_ref"] = distances
        # Dummy
        trip["route_dummy"] = 1.0
        return trip

# --------------------------------------------------------------------------- #
# Model Loading and Scoring
# --------------------------------------------------------------------------- #
class Scorer:
    def __init__(self, dispatcher_path: Path):
        self.dispatcher = self._load_dispatcher(dispatcher_path)

    def _load_dispatcher(self, path: Path) -> Dict[Any, str]:
        try:
            return joblib.load(path)
        except Exception as e:
            logging.error("Failed to load dispatcher: %s", e)
            sys.exit(1)

    def load_artifacts(self, route: Any) -> Dict[str, Any]:
        model_path = self.dispatcher.get(route)
        if not model_path:
            raise KeyError(f"No model for route {route}")
        return joblib.load(model_path)

    def score(self, artefacts: Dict[str, Any], X_raw: np.ndarray) -> Any:
        tau = artefacts["tau"]
        pipe = artefacts.get("pipeline")
        if pipe is not None:
            if hasattr(pipe, "predict_proba"):
                scores = pipe.predict_proba(X_raw)[:,1]
            else:
                scores = -pipe.decision_function(X_raw)
        else:
            scaler: StandardScaler = artefacts["scaler"]
            model = artefacts["model"]
            Xs = scaler.transform(X_raw)
            if hasattr(model, "predict_proba"):
                scores = model.predict_proba(Xs)[:,1]
            else:
                scores = -model.decision_function(Xs)
        preds = (scores > tau).astype(int)
        return scores, preds

# --------------------------------------------------------------------------- #
# CLI and Execution
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score a single trip and dump JSON of anomalies"
    )
    parser.add_argument("trip_id", help="Identifier of the trip to score")
    parser.add_argument("-i", "--input",
                        type=Path,
                        default=Path("clean_data/all_anomalies_combined.parquet"),
                        help="Input Parquet file path")
    parser.add_argument("-o", "--output",
                        type=Path,
                        default=Path("trip.json"),
                        help="Output JSON file path")
    parser.add_argument("-d", "--dispatcher",
                        type=int,
                        choices=list(DISPATCHER_PATHS.keys()),
                        required=True,
                        help="Select dispatcher: 1=OC-SVM, 2=Isolation Forest, 3=Logistic Regression, 4=Random Forest"
                        )
    parser.add_argument("--log-level",
                        choices=["DEBUG","INFO","WARNING","ERROR","CRITICAL"],
                        default="INFO",
                        help="Logging level"
                        )
    return parser.parse_args()


def setup_logging(level: str) -> None:
    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(message)s",
        level=getattr(logging, level)
    )


def build_output_json(df: pd.DataFrame, scores: np.ndarray, preds: np.ndarray) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for idx, row in df.reset_index(drop=True).iterrows():
        data_fields = {
            k: (row[k].isoformat() if hasattr(row[k], 'isoformat') else row[k])
            # [
            #     'trip_id','start_latitude','start_longitude','start_time',
            #     'end_latitude','end_longitude','end_time','start_port',
            #     'end_port','ship_type','length','breadth','draught',
            #     'speed_over_ground','course_over_ground','true_heading',
            #     'destination','is_anomaly','dv','dcourse','ddraft',
            #     'zone','x_km','y_km','dist_to_ref','route_dummy'
            # ]
            for k in [
                'ship_type','length','breadth','draught',
                'speed_over_ground','course_over_ground','true_heading',
                'dv','dcourse','ddraft',
                'zone','x_km','y_km','dist_to_ref'
            ] if k in df.columns
        }
        out.append({
            "latitude": float(row.latitude),
            "longitude": float(row.longitude),
            "time_stamp": row.time_stamp.isoformat() if pd.notna(row.time_stamp) else None,
            "score": float(scores[idx]),
            "is_anomaly_pred": int(preds[idx]),
            "data": data_fields
        })
    return out


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)

    dispatcher_file = Path(DISPATCHER_PATHS[args.dispatcher])
    logging.info("Using dispatcher from choice %d -> %s", args.dispatcher, dispatcher_file)

    try:
        df_all = pd.read_parquet(args.input, engine="pyarrow")
    except Exception as e:
        logging.error("Failed to read input: %s", e)
        sys.exit(1)

    processor = DataProcessor(df_all)
    try:
        trip_df = processor.get_trip(args.trip_id)
    except ValueError as e:
        logging.error(e)
        sys.exit(1)

    feature_df = processor.build_feature_frame(trip_df)

    route = feature_df.start_port.iloc[0]
    scorer = Scorer(dispatcher_file)
    try:
        artefacts = scorer.load_artifacts(route)
    except KeyError as e:
        logging.error(e)
        sys.exit(1)

    feats = artefacts.get("features", [])
    X_raw = feature_df[feats].fillna(0).values
    scores, preds = scorer.score(artefacts, X_raw)

    output_data = build_output_json(feature_df, scores, preds)
    try:
        args.output.write_text(json.dumps(output_data, ensure_ascii=False, indent=2))
        logging.info("✓ %d points scored. JSON saved → %s", len(output_data), args.output)
    except Exception as e:
        logging.error("Failed writing JSON: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
