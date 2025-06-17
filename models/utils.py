import numpy as np
import plotly.graph_objects as go


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
