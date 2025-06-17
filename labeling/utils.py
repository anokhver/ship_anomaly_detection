import numpy as np
from matplotlib import pyplot as plt
import contextily as ctx
from pyproj import Transformer
import plotly.graph_objects as go
from matplotlib.cm import get_cmap


def visualize_trajectory_clusters_interactive_fancy(compressed_trajectories, trip_ids, cluster_labels):
    """
    Fixed version with map display that properly shows all trajectories while being performant
    """
    print("\n--- Optimized Trajectory Clusters Visualization ---")

    # Get unique cluster labels
    unique_labels = sorted(set(cluster_labels))
    colors = get_cmap('tab20', len(unique_labels))  # Using tab20 for more colors

    # Prepare data
    fig = go.Figure()

    # First pass to get bounds
    all_lats = []
    all_lons = []
    for trip_id in trip_ids:
        trajectory = compressed_trajectories[trip_id]
        all_lats.extend(trajectory[:, 0])
        all_lons.extend(trajectory[:, 1])

    # Calculate center and zoom level (approximate)
    center_lat = np.mean(all_lats)
    center_lon = np.mean(all_lons)
    lat_range = max(all_lats) - min(all_lats)
    lon_range = max(all_lons) - min(all_lons)
    zoom = 11 - max(lat_range, lon_range)  # Adjust this heuristic as needed

    # Create one trace per cluster for better performance
    for label in unique_labels:
        mask = np.array(cluster_labels) == label
        cluster_trips = [trip_ids[i] for i in range(len(trip_ids)) if mask[i]]

        # Combine all trajectories in this cluster with None separators
        segments_lats = []
        segments_lons = []
        hover_texts = []

        for trip_id in cluster_trips:
            traj = compressed_trajectories[trip_id]
            segments_lats.extend(traj[:, 0])
            segments_lons.extend(traj[:, 1])
            segments_lats.append(None)  # Break between trajectories
            segments_lons.append(None)
            # Add hover text for each point (optional)
            hover_texts.extend([f'Trip: {trip_id}<br>Cluster: {label}'] * len(traj))
            hover_texts.append(None)

        color = f'rgb{colors(label)[:3]}' if label != -1 else 'gray'
        name = f'Cluster {label}' if label != -1 else 'Noise'
        line_width = 1 if label == -1 else 2
        opacity = 0.7 if label == -1 else 0.9

        fig.add_trace(go.Scattermapbox(
            lat=segments_lats,
            lon=segments_lons,
            mode='lines',
            line=dict(width=line_width, color=color),
            opacity=opacity,
            name=name,
            hoverinfo='text',
            hovertext=hover_texts,
            showlegend=True
        ))

    fig.update_layout(
        mapbox_style="open-street-map",  # Or "open-street-map", "carto-positron"
        mapbox=dict(
            center=dict(lat=center_lat, lon=center_lon),
            zoom=zoom,
        ),
        margin={"r": 0, "t": 30, "l": 0, "b": 0},
        title='Ship Trajectory Clusters',
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )
    print("--- Visualization complete. Use the interactive map to explore clusters.")
    return fig


def visualize_trip_interactive(df, trip_id):
    """
    Visualizes a single trip from the DataFrame interactively using Plotly.

    Args:
        df (pd.DataFrame): The DataFrame containing all trip data.
        trip_id (int): The ID of the trip to visualize.
    """
    print(f"\n--- Visualizing Trip ID: {trip_id} (Interactive Map) ---")

    # 1. Filter the DataFrame to get data for the specified trip
    trip_df = df[df['trip_id'] == trip_id].sort_values(by='time_stamp')

    # 2. Check if the trip was found
    if trip_df.empty:
        print(f"Error: Trip ID {trip_id} not found in the DataFrame.")
        return

    # 3. Extract coordinates for plotting
    longitudes = trip_df['longitude'].tolist()
    latitudes = trip_df['latitude'].tolist()
    timestamps = trip_df['time_stamp'].dt.strftime('%Y-%m-%d %H:%M:%S').tolist()  # Format time for hover info

    # Get start and end points for highlighting
    start_point_lon, start_point_lat = longitudes[0], latitudes[0]
    end_point_lon, end_point_lat = longitudes[-1], latitudes[-1]

    # Calculate center and an approximate zoom level for the map
    center_lat = np.mean(latitudes)
    center_lon = np.mean(longitudes)

    # Simple heuristic for zoom: adjust based on the spread of coordinates
    # You might need to fine-tune this for your specific data
    lat_range = max(latitudes) - min(latitudes)
    lon_range = max(longitudes) - min(longitudes)

    # A base zoom level, then subtract based on how spread out the points are
    # Smaller values for range mean higher zoom (closer in)
    zoom = 10
    if lat_range > 0 and lon_range > 0:
        zoom = 12 - np.log(max(lat_range, lon_range))  # Logarithmic scaling can work well
    zoom = max(1, min(zoom, 15))  # Keep zoom within reasonable bounds

    # 4. Create the Plotly figure
    fig = go.Figure()

    # Add the trip trajectory path
    # Using 'lines' mode is more efficient for trajectories than 'lines+markers'
    fig.add_trace(go.Scattermapbox(
        mode="lines",
        lon=longitudes,
        lat=latitudes,
        hoverinfo='text',
        text=[f'Time: {t}<br>Lat: {lat:.4f}<br>Lon: {lon:.4f}'
              for t, lat, lon in zip(timestamps, latitudes, longitudes)],
        line=dict(width=4, color='blue'),
        name=f'Trip Path (ID: {trip_id})'
    ))

    # Add start point marker
    fig.add_trace(go.Scattermapbox(
        mode="markers",
        lon=[start_point_lon],
        lat=[start_point_lat],
        marker=dict(
            size=12,
            color='green',
            symbol='circle',
        ),
        name='Start Point',
        hoverinfo='text',
        text=[f'Start: {timestamps[0]}']
    ))

    # Add end point marker
    fig.add_trace(go.Scattermapbox(
        mode="markers",
        lon=[end_point_lon],
        lat=[end_point_lat],
        marker=dict(
            size=12,
            color='red',
            symbol='circle',
        ),
        name='End Point',
        hoverinfo='text',
        text=[f'End: {timestamps[-1]}']
    ))

    # 5. Finalize and display the plot
    fig.update_layout(
        title=f'Interactive Visualization of Trip ID: {trip_id}',
        mapbox_style="open-street-map",  # You can choose other styles like "carto-positron", "stamen-terrain", etc.
        mapbox=dict(
            center=dict(lat=center_lat, lon=center_lon),
            zoom=zoom,
        ),
        margin={"r": 0, "t": 50, "l": 0, "b": 0},
        hovermode='closest',  # Shows hover info for the closest point
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )

    print(f"Return interactive map for Trip ID {trip_id}.")
    return fig
