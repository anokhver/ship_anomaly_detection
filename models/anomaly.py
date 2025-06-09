import pandas as pd
import numpy as np
from geopy.distance import geodesic

gps_stop_distance_km     = 10
course_change_threshold  = 30
speed_change_threshold   = 3
draught_change_threshold = 0.5
distance_anomaly_km      = 3

print("1) Loading data")
df = pd.read_parquet('from_KIEL.parquet')

print("2) Sorting and computing deltas")
df = df.sort_values(['trip_id', 'time_stamp'])
for col in ['speed_over_ground', 'course_over_ground', 'draught']:
    df[f'delta_{col}'] = df.groupby('trip_id')[col].diff().abs()

print("3) Computing delta_pos_km")
df['prev_latitude']  = df.groupby('trip_id')['latitude'].shift(1)
df['prev_longitude'] = df.groupby('trip_id')['longitude'].shift(1)

def haversine_pair(lat1, lon1, lat2, lon2):
    if pd.isnull(lat2) or pd.isnull(lon2):
        return np.nan
    return geodesic((lat1, lon1), (lat2, lon2)).kilometers

df['delta_pos_km'] = df.apply(
    lambda r: haversine_pair(r.latitude, r.longitude, r.prev_latitude, r.prev_longitude),
    axis=1
)
df = df.drop(columns=['prev_latitude', 'prev_longitude'])

print("4) Detecting long stops")
ports = (
    df[['trip_id','start_latitude','start_longitude','end_latitude','end_longitude']]
    .drop_duplicates('trip_id')
    .set_index('trip_id')
)
def is_long_stop(row):
    start = (ports.loc[row.trip_id, 'start_latitude'],
             ports.loc[row.trip_id, 'start_longitude'])
    end   = (ports.loc[row.trip_id, 'end_latitude'],
             ports.loc[row.trip_id, 'end_longitude'])
    ds = geodesic((row.latitude, row.longitude), start).kilometers
    de = geodesic((row.latitude, row.longitude), end).kilometers
    return (row.speed_over_ground < 0.5) and (ds > gps_stop_distance_km) and (de > gps_stop_distance_km)

df['anomaly_stop'] = df.apply(is_long_stop, axis=1)

print("5) Detecting drastic changes in course, speed, and draught")
df['anomaly_course']  = df['delta_course_over_ground'] > course_change_threshold
df['anomaly_speed']   = df['delta_speed_over_ground']  > speed_change_threshold
df['anomaly_draught'] = df['delta_draught']            > draught_change_threshold

print("6) Detecting large movement (>3 km)")
df['anomaly_distance'] = df['delta_pos_km'] > distance_anomaly_km

print("7) Combining all flags into is_anomaly")
df['is_anomaly'] = df[
    ['anomaly_stop','anomaly_course','anomaly_speed','anomaly_draught','anomaly_distance']
].any(axis=1)

print("8) Converting dates to strings and saving results")
for col in ['start_time', 'end_time', 'time_stamp']:
    df[col] = pd.to_datetime(df[col]).dt.strftime('%Y-%m-%d %H:%M:%S')

df.to_csv('kiel_anomalies.csv', index=False)
df.to_excel('kiel_anomalies.xlsx', index=False)

print("Done")
