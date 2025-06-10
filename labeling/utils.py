import numpy as np
from matplotlib import pyplot as plt
import contextily as ctx
from pyproj import Transformer
import plotly.graph_objects as go
from matplotlib.cm import get_cmap


def create_artificial_anomaly(df, trip_id, anomaly_type='position', start_index=50, duration=10):
    """
    Injects an artificial anomaly into a specific trip within the DataFrame.

    Args:
        df (pd.DataFrame): The main DataFrame containing all trips.
        trip_id (int): The ID of the trip to modify.
        anomaly_type (str): Type of anomaly ('position', 'speed', 'stop', 'course').
        start_index (int): The row index within the trip where the anomaly starts.
        duration (int): How many data points the anomaly should last for.

    Returns:
        pd.DataFrame: A new DataFrame with the anomalous trip.
    """
    # Create a copy to avoid modifying the original DataFrame
    anomalous_df = df.copy()

    # Get the indices for the target trip
    trip_indices = anomalous_df[anomalous_df['trip_id'] == trip_id].sort_values("time").index

    # Define the slice where the anomaly will occur
    if len(trip_indices) < start_index + duration:
        print("Warning: Trip is too short for the specified anomaly. No changes made.")
        return df

    anomaly_slice = trip_indices[start_index: start_index + duration]

    print(f"Injecting a '{anomaly_type}' anomaly into Trip {trip_id} from index {start_index} for {duration} points.")

    if anomaly_type == 'position':
        # Simulate deviating from the route by adding an offset
        anomalous_df.loc[anomaly_slice, 'Latitude'] += np.random.uniform(-0.1, 0.1, size=len(anomaly_slice))
        anomalous_df.loc[anomaly_slice, 'Longitude'] -= np.random.uniform(-0.1, 0.1, size=len(anomaly_slice))

    elif anomaly_type == 'speed':
        # Simulate a sudden, unnatural speed increase
        anomalous_df.loc[anomaly_slice, 'SOG'] = 30.0  # Set to a high speed

    elif anomaly_type == 'stop':
        # Simulate an abnormal stop in the middle of a channel
        anomalous_df.loc[anomaly_slice, 'SOG'] = 0.0

    elif anomaly_type == 'course':
        # Simulate abnormal steering where COG doesn't match the path
        # If the ship is moving, but its course is suddenly backwards
        anomalous_df.loc[anomaly_slice, 'COG'] = 180.0

    return anomalous_df


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


def visualize_trajectory_clusters(compressed_trajectories, trip_ids, cluster_labels):
    """
    Visualizes trajectory clusters on a schematic map background.

    Parameters:
    - compressed_trajectories: A dictionary mapping TripID to its [lat, lon] points
    - trip_ids: A list of the TripIDs
    - cluster_labels: A list/array of the cluster label for each corresponding trajectory

    Returns:
    - A matplotlib plot object with the trajectories colored by their cluster labels.
    """
    print("\n--- Visualizing Trajectory Clusters with Schematic Map Background ---")

    # Create a figure and axes for the plot
    fig, ax = plt.subplots(figsize=(12, 10))

    # Get the unique cluster labels (e.g., -1, 0, 1, 2)
    unique_labels = set(cluster_labels)

    # Generate a set of unique colors for each cluster
    colors = plt.cm.get_cmap('tab10', len(unique_labels))

    # Keep track of labels that have been added to the legend
    legend_labels = set()

    # Initialize coordinate bounds for map extent
    min_lat, max_lat = float('inf'), float('-inf')
    min_lon, max_lon = float('inf'), float('-inf')

    # First pass: collect all coordinates to determine map bounds
    all_trajectories = []
    for i, trip_id in enumerate(trip_ids):
        trajectory_points = compressed_trajectories[trip_id]
        all_trajectories.append((trajectory_points, cluster_labels[i]))

        # Update bounds
        min_lat = min(min_lat, trajectory_points[:, 0].min())
        max_lat = max(max_lat, trajectory_points[:, 0].max())
        min_lon = min(min_lon, trajectory_points[:, 1].min())
        max_lon = max(max_lon, trajectory_points[:, 1].max())

    # Add some padding to the bounds
    lat_padding = (max_lat - min_lat) * 0.1
    lon_padding = (max_lon - min_lon) * 0.1
    min_lat -= lat_padding
    max_lat += lat_padding
    min_lon -= lon_padding
    max_lon += lon_padding

    # Set up coordinate transformation (WGS84 to Web Mercator for contextily)
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)

    # Transform bounds to Web Mercator
    west_merc, south_merc = transformer.transform(min_lon, min_lat)
    east_merc, north_merc = transformer.transform(max_lon, max_lat)

    # Loop through each trajectory and its assigned cluster label
    for trajectory_points, label in all_trajectories:
        # Transform coordinates to Web Mercator for plotting
        lon_merc, lat_merc = transformer.transform(trajectory_points[:, 1], trajectory_points[:, 0])

        # Noise points (label -1) are plotted in gray and without a legend entry
        if label == -1:
            color = 'gray'
            plot_label = None  # No legend for noise
            linewidth = 1.5
            alpha = 0.6
        else:
            # Assign a unique color to each cluster
            color = colors(label)
            plot_label = f'Cluster {label}'
            linewidth = 2
            alpha = 0.8

        # Add to legend only once per cluster
        if plot_label and plot_label not in legend_labels:
            ax.plot(lon_merc, lat_merc, color=color, label=plot_label,
                    alpha=alpha, linewidth=linewidth)
            legend_labels.add(plot_label)
        else:
            ax.plot(lon_merc, lat_merc, color=color, alpha=alpha, linewidth=linewidth)

    # Set the map extent
    ax.set_xlim(west_merc, east_merc)
    ax.set_ylim(south_merc, north_merc)

    # Add a schematic map background (trying multiple options)
    try:
        # Option 1: OpenStreetMap standard (clean schematic)
        ctx.add_basemap(ax, crs="EPSG:3857", source=ctx.providers.OpenStreetMap.Mapnik, alpha=0.8)
    except:
        try:
            # Option 2: Stamen Toner (high-contrast black and white)
            ctx.add_basemap(ax, crs="EPSG:3857", source=ctx.providers.Stamen.TonerLite, alpha=0.8)
        except:
            try:
                # Option 3: CartoDB Positron (light theme)
                ctx.add_basemap(ax, crs="EPSG:3857", source=ctx.providers.CartoDB.Positron, alpha=0.8)
            except:
                # Final fallback: Simple grayscale
                ctx.add_basemap(ax, crs="EPSG:3857", source=ctx.providers.CartoDB.VoyagerNoLabels, alpha=0.8)

    # --- Finalize and show the plot ---
    ax.set_title('Ship Trajectories by Cluster', fontsize=14, fontweight='bold')
    ax.set_xlabel('Longitude (Web Mercator)', fontsize=12)
    ax.set_ylabel('Latitude (Web Mercator)', fontsize=12)

    # Position legend outside the plot area
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    # Remove grid as it may interfere with map visibility
    ax.grid(False)

    # Adjust layout to prevent legend cutoff
    plt.tight_layout()

    print("Displaying plot with schematic map background... Close the plot window to continue.")
    return plt


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


def visualize_trip(df, trip_id):
    """
    Visualizes a single trip from the DataFrame based on its TripID.

    Args:
        df (pd.DataFrame): The DataFrame containing all trip data.
        trip_id (int): The ID of the trip to visualize.
    """
    # --- 1. Filter the DataFrame to get data for the specified trip ---
    trip_df = df[df['trip_id'] == trip_id].sort_values(by='time')
    # trip_df = test_df.sort_values(by='time')

    # --- 2. Check if the trip was found ---
    if trip_df.empty:
        print(f"Error: Trip ID {trip_id} not found in the DataFrame.")
        return

    # --- 3. Extract coordinates for plotting ---
    longitudes = trip_df['Longitude']
    latitudes = trip_df['Latitude']

    # Get start and end points for highlighting
    start_point = (longitudes.iloc[0], latitudes.iloc[0])
    end_point = (longitudes.iloc[-1], latitudes.iloc[-1])

    # --- 4. Create the plot ---
    fig, ax = plt.subplots(figsize=(12, 9))

    # Plot the trajectory path
    ax.plot(longitudes, latitudes, 'b-', label='Trip Path', zorder=1)

    # Highlight the start point in green
    ax.scatter(start_point[0], start_point[1], c='green', s=100, label='Start Point', zorder=5, edgecolors='black')

    # Highlight the end point in red
    ax.scatter(end_point[0], end_point[1], c='red', s=100, label='End Point', zorder=5, edgecolors='black')

    # Add text labels for start and end
    ax.text(start_point[0], start_point[1], ' Start', va='center')
    ax.text(end_point[0], end_point[1], ' End', va='center')

    # --- 5. Finalize and display the plot ---
    ax.set_title(f'Visualization of Trip ID: {trip_id}')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.legend()
    ax.grid(True)

    print(f"Displaying plot for Trip ID {trip_id}. Close the plot window to exit.")
    plt.show()

