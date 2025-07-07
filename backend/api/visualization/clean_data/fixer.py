#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Trip-wise anomaly labeller with undo confirmation and detailed logging

*   Anomalies = (manual intervals or Δ-speed>thr or Δ-course>thr) \ ZONES
*   All other trip points → no anomaly.
*   Before writing: displays override statistics and requires confirmation.
"""
import argparse
import json
import os
import sys
from typing import List, Tuple

import numpy as np
import pandas as pd

# ------------  CONFIGURATION  -------------------------------------------------
ZONES = [
    [53.8, 53.5,  8.60,  8.14],
    [53.66, 53.0, 11.00,  9.50],
    [54.45, 54.2, 10.30, 10.00],
    [54.71, 54.25,19.00, 18.35],
]
DEFAULT_SPEED_THR  = 3.0   # [knots]
DEFAULT_COURSE_THR = 30.0  # [degrees]
# -----------------------------------------------------------------------------


def parse_intervals(str_intervals: List[str]) -> List[Tuple[int, int]]:
    """Convert ["10-20","55-60"] → [(10,20),(55,60)], ignoring invalid entries."""
    intervals = []
    for s in str_intervals:
        try:
            start, end = map(int, s.split("-"))
            intervals.append((min(start, end), max(start, end)))
        except Exception:
            print(f"Ignoring invalid interval: '{s}'", file=sys.stderr)
    return intervals


def point_in_zones(lat: float, lon: float) -> bool:
    """Return True if (lat, lon) lies within any of the defined ZONES."""
    for lat_max, lat_min, lon_max, lon_min in ZONES:
        if lat_min <= lat <= lat_max and lon_min <= lon <= lon_max:
            return True
    return False


def confirm(prompt: str) -> bool:
    """Ask user for yes/no confirmation."""
    answer = input(f"{prompt} [y/N]: ").strip().lower()
    return answer in ('y', 'yes')


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Labels anomalies for a single trip with confirmation before saving."
        )
    )
    parser.add_argument("trip_id", help="Trip identifier to process (int or str)")
    parser.add_argument("-i", "--input", default="all_anomalies_combined.parquet",
                        help="Input Parquet file")
    parser.add_argument("-o", "--output", default="all_anomalies_combined.parquet",
                        help="Output Parquet file for saving changes (can overwrite input)")
    parser.add_argument("--export-trip-json", default=".",
                        help="Export trip points to JSON ('.'→trip.json)")
    parser.add_argument("-d", "--interval", nargs="+", default=[],
                        help="Index intervals to mark as anomalies, format 'start-end'")
    parser.add_argument("--speed-thr", type=float, default=DEFAULT_SPEED_THR,
                        help=f"Δ-speed threshold in knots (default {DEFAULT_SPEED_THR})")
    parser.add_argument("--course-thr", type=float, default=DEFAULT_COURSE_THR,
                        help=f"Δ-course threshold in degrees (default {DEFAULT_COURSE_THR})")
    args = parser.parse_args()

    # Load the entire dataset (we will modify only the specified trip)
    try:
        df = pd.read_parquet(args.input, engine="pyarrow")
    except Exception as e:
        print(f"Error reading Parquet file: {e}", file=sys.stderr)
        sys.exit(1)

    # Ensure is_anomaly column exists and is BooleanDtype
    if "is_anomaly" not in df.columns:
        df["is_anomaly"] = False
    df["is_anomaly"] = df["is_anomaly"].astype("boolean")

    # Determine trip identifier type
    col = df["trip_id"]
    trip_id = int(args.trip_id) if col.dtype.kind in "iu" else args.trip_id
    trip = df[df["trip_id"] == trip_id].copy()
    if trip.empty:
        print(f"No points found for trip_id={args.trip_id}", file=sys.stderr)
        sys.exit(1)
    trip = trip.sort_values("time_stamp")
    print(f"1) Loaded trip {args.trip_id} with {len(trip)} points.")

    # Compute Δ-speed and Δ-course
    trip["delta_speed"] = trip["speed_over_ground"].diff().abs()
    course_diff = trip["course_over_ground"].diff().abs()
    trip["delta_course"] = course_diff.where(course_diff <= 180, 360 - course_diff)

    # Manual intervals anomaly flags
    manual_flags = pd.Series(False, index=trip.index)
    for start, end in parse_intervals(args.interval):
        manual_flags.iloc[start-1:end] = True
    print(f"2) Manual intervals flagged: {manual_flags.sum()} points.")

    # Δ-threshold anomaly flags
    delta_flags = (
        (trip["delta_speed"] > args.speed_thr) |
        (trip["delta_course"] > args.course_thr)
    )
    print(f"3) Δ-thresholds detected: {delta_flags.sum()} points.")

    # Combine manual and delta-based flags
    combined_flags = manual_flags | delta_flags
    print(f"4) Combined before zone override: {combined_flags.sum()} points.")

    # Override flags within zones
    in_zone_mask = trip.apply(lambda r: point_in_zones(r.latitude, r.longitude), axis=1)
    overridden_count = (combined_flags & in_zone_mask).sum()
    print(f"5) Overridden in zones: {overridden_count} points.")

    # Final anomaly flags (excluding zone points)
    final_flags = combined_flags & ~in_zone_mask
    print(f"6) Final anomalies flagged: {final_flags.sum()} points.")

    # Export trip JSON preview
    if args.export_trip_json is not None:
        export_path = (
            os.path.join(os.getcwd(), "trip.json")
            if args.export_trip_json == "."
            else args.export_trip_json
        )
        os.makedirs(os.path.dirname(export_path) or ".", exist_ok=True)
        payload = [
            {
                "latitude": float(row.latitude),
                "longitude": float(row.longitude),
                "is_anomaly_pred": int(final_flags.loc[idx]),
                "comment": f"Entry number {i+1}"
            }
            for i, (idx, row) in enumerate(trip.iterrows())
        ]
        with open(export_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"Preview JSON saved to: {export_path}")

    # Confirm before writing changes
    if not confirm("Save modified Parquet file?"):
        print("Save canceled. No changes were written.")
        sys.exit(0)

    # Write back anomaly flags to original DataFrame
    df.loc[trip.index, "is_anomaly"] = final_flags.astype("boolean")
    if "y_true" in df.columns:
        df.loc[trip.index, "y_true"] = final_flags.astype("int8")

    # Save to Parquet
    try:
        df.to_parquet(args.output, engine="pyarrow", index=False)
        print(f"✓ Modified file saved to: {args.output}")
    except Exception as e:
        print(f"Error saving Parquet file: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
