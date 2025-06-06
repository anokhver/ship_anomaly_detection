import argparse
import csv
import json
import sys

def extract_trip_points(input_csv_path: str, trip_id: str):
    """
    Usage:
    python data_visualizer.py TripID
    """
    points = []

    with open(input_csv_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if row.get('TripID') == trip_id:
                try:
                    lat = float(row['Latitude'])
                    lon = float(row['Longitude'])
                except (KeyError, ValueError):
                    continue

                points.append({
                    'latitude': lat,
                    'longitude': lon
                })

    return points

def main():
    parser = argparse.ArgumentParser(
        description="Extracts all points (Latitude, Longitude) for a given TripID from the CSV file and saves them as JSON."
    )
    parser.add_argument(
        'tripid',
    )
    parser.add_argument(
        '--input',
        '-i',
        default='cleaned.csv',
    )
    parser.add_argument(
        '--output',
        '-o',
        default='/workspace/frontend/src/assets/trip.json',
    )

    args = parser.parse_args()

    trip_id = args.tripid
    input_csv = args.input
    output_json = args.output

    points = extract_trip_points(input_csv, trip_id)

    if not points:
        print(f"No points found for TripID={trip_id}.", file=sys.stderr)
    else:
        print(f"Found {len(points)} points for TripID={trip_id}.")

    with open(output_json, 'w', encoding='utf-8') as jsonfile:
        json.dump(points, jsonfile, ensure_ascii=False, indent=2)

    print(f"Saved to file: {output_json}")

if __name__ == '__main__':
    main()
