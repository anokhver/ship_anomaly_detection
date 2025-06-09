import argparse
import json
import sys
import pandas as pd  # Add pandas for Parquet support

def extract_trip_points_parquet(input_parquet_path: str, trip_id: str):
    """
    Extracts all points (Latitude, Longitude) for a given TripID from a Parquet file,
    ordered by time_stamp ascending.
    """
    points = []

    try:
        df = pd.read_parquet(input_parquet_path)
    except Exception as e:
        print(f"Error reading Parquet file: {e}", file=sys.stderr)
        return points
    print(df['trip_id'].unique())

    trip_id_col_type = df['trip_id'].dtype
    if trip_id_col_type == 'int64' or trip_id_col_type == 'int32':
        trip_id = int(trip_id)
    filtered = df[df['trip_id'] == trip_id]

    # Order by time_stamp ascending
    if 'time_stamp' in filtered.columns:
        filtered = filtered.sort_values('time_stamp', ascending=True)

    for idx, (_, row) in enumerate(filtered.iterrows(), 1):
        try:
            lat = float(row['latitude'])
            lon = float(row['longitude'])
            points.append({
                'latitude': lat,
                'longitude': lon,
                'comment': f'Entry number {idx}'
            })
        except (KeyError, ValueError, TypeError):
            continue

    return points

def main():
    parser = argparse.ArgumentParser(
        description="Extracts all points (Latitude, Longitude) for a given TripID from a Parquet file and saves them as JSON."
    )
    parser.add_argument(
        'trip_id',
    )
    parser.add_argument(
        '--input',
        '-i',
        default='from_KIEL.parquet',
        help='Input Parquet file'
    )
    parser.add_argument(
        '--output',
        '-o',
        default='/workspace/frontend/src/assets/trip.json',
    )

    args = parser.parse_args()

    trip_id = args.trip_id
    input_parquet = args.input
    output_json = args.output

    points = extract_trip_points_parquet(input_parquet, trip_id)

    if not points:
        print(f"No points found for TripID={trip_id}.", file=sys.stderr)
    else:
        print(f"Found {len(points)} points for TripID={trip_id}.")

    with open(output_json, 'w', encoding='utf-8') as jsonfile:
        json.dump(points, jsonfile, ensure_ascii=False, indent=2)

    print(f"Saved to file: {output_json}")

if __name__ == '__main__':
    main()