import numpy as np
import pandas as pd
from scipy.spatial.distance import euclidean
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score
from sklearn.ensemble import HistGradientBoostingRegressor
from typing import List, Optional, Union, Any
import warnings


def plot_missing(df: pd.DataFrame) -> sns.matrix.ClusterGrid:
    """
    Create a heatmap visualization of missing values in the dataframe.

    Args:
        df: Input dataframe

    Returns:
        Seaborn heatmap object
    """
    cols = df.columns
    plt.figure(figsize=(12, 6))
    return sns.heatmap(
        df[cols].isnull(),
        cmap=['white', 'black'],
        cbar=False,
        xticklabels=True,
        yticklabels=False
    )


def get_percentage_missing(df: pd.DataFrame, col: Optional[str] = None) -> Union[float, pd.Series]:
    """
    Calculate percentage of missing values for a column or all columns.

    Args:
        df: Input dataframe
        col: Column name (optional). If None, returns for all columns

    Returns:
        Percentage of missing values
    """
    if col is None:
        return (df.isnull().sum() / len(df) * 100).round(5)
    else:
        if col not in df.columns:
            warnings.warn(f"Column '{col}' not found in dataframe")
            return 0.0
        return round(df[col].isnull().sum() / len(df) * 100, 5)


def get_entries_with_missing_values(df: pd.DataFrame, col: str) -> pd.Series:
    """
    Get unique values for each trip that has missing values in the specified column.

    Args:
        df: Input dataframe
        col: Column to check for missing values

    Returns:
        Series with TripID as index and list of unique values as values
    """
    if col not in df.columns:
        raise ValueError(f"Column '{col}' not found in dataframe")

    if 'TripID' not in df.columns:
        raise ValueError("TripID column not found in dataframe")

    # Get TripIDs with at least one missing value
    trips_with_missing = df[df[col].isna()]['TripID'].unique()

    if len(trips_with_missing) == 0:
        return pd.Series(dtype=object, name=col)

    # Filter DataFrame to include only these trips
    df_missing_trips = df[df['TripID'].isin(trips_with_missing)]

    return (
        df_missing_trips
        .groupby('TripID')[col]
        .apply(lambda x: list(x.unique()))
    )


def get_inconsistent_trip_ids(df: pd.DataFrame, column: str) -> pd.Index:
    """
    Identify TripIDs that have inconsistent values in the specified column.

    Args:
        df: Input dataframe
        column: Column to check for inconsistencies

    Returns:
        Index of inconsistent TripIDs
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in dataframe")

    if 'TripID' not in df.columns:
        raise ValueError("TripID column not found in dataframe")

    # Group by TripID and analyze consistency
    grouped = df.groupby("TripID")[column]

    # Count unique non-null values per trip
    nunique_values = grouped.nunique()

    # Check if trip has any NaN values
    has_nan_values = grouped.apply(lambda x: x.isna().any())

    # A trip is inconsistent if:
    # 1. It has more than 1 unique value, OR
    # 2. It has both NaN and non-NaN values
    inconsistent_by_nunique = nunique_values > 1
    inconsistent_by_nan_and_value = has_nan_values & (nunique_values > 0)

    inconsistent_trip_ids = (inconsistent_by_nunique | inconsistent_by_nan_and_value)

    return inconsistent_trip_ids.loc[lambda x: x].index


def make_inconsistent_mode(dataf: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    Fill inconsistent values in a column with the mode value for each trip.

    Args:
        dataf: Input dataframe (modified in place)
        column: Column to process

    Returns:
        Modified dataframe
    """
    if column not in dataf.columns:
        raise ValueError(f"Column '{column}' not found in dataframe")

    inconsistent_trip_ids = get_inconsistent_trip_ids(dataf, column)

    if len(inconsistent_trip_ids) == 0:
        print(f"No inconsistent trips found for column '{column}'")
        return dataf

    print(f"Processing {len(inconsistent_trip_ids)} inconsistent trips for column '{column}'")

    for trip_id in inconsistent_trip_ids:
        trip_mask = dataf["TripID"] == trip_id
        trip_values = dataf.loc[trip_mask, column]

        # Calculate mode excluding NaN values
        mode_values = trip_values.mode(dropna=True)

        if not mode_values.empty:
            chosen_mode = mode_values.iloc[0]
            # Replace ALL values (including nulls) with the first mode
            dataf.loc[trip_mask, column] = chosen_mode
        else:
            print(f"Warning: No mode found for TripID {trip_id} in column '{column}'")

    return dataf


def all_fill_with_mode(dataf: pd.DataFrame, column: str) -> None:
    """
    Fill missing values in a column using mode-based approach with detailed reporting.

    Args:
        dataf: Input dataframe (modified in place)
        column: Column to process
    """
    if column not in dataf.columns:
        raise ValueError(f"Column '{column}' not found in dataframe")

    print("-" * 50)
    print(f"PROCESSING COLUMN: {column}")
    print("-" * 50)

    # Before processing
    missing_before = get_percentage_missing(dataf, column)
    inconsistent_before = get_inconsistent_trip_ids(dataf, column)

    print(f"Before processing:")
    print(f"  - Missing values: {missing_before:.2f}%")
    print(f"  - Inconsistent trips: {len(inconsistent_before)}")

    # Process the data
    make_inconsistent_mode(dataf, column)

    # After processing
    missing_after = get_percentage_missing(dataf, column)
    inconsistent_after = get_inconsistent_trip_ids(dataf, column)

    print(f"\nAfter processing:")
    print(f"  - Missing values: {missing_after:.2f}%")
    print(f"  - Inconsistent trips: {len(inconsistent_after)}")
    print(f"  - Improvement: {missing_before - missing_after:.2f}% reduction in missing values")
    print("-" * 50)


def fill_missing(
        df: pd.DataFrame,
        target_col: str,
        feature_cols: List[str],
        round_values: bool = True
) -> pd.DataFrame:
    """
    Fill missing values using machine learning regression.

    Args:
        df: Input dataframe
        target_col: Column with missing values to fill
        feature_cols: Columns to use as features for prediction
        round_values: Whether to round predicted values to integers

    Returns:
        Dataframe with filled missing values
    """
    print(f"\n\n Fill missing values using machine learning regression in column '{target_col}' using features: {feature_cols}")

    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataframe")

    missing_features = [col for col in feature_cols if col not in df.columns]
    if missing_features:
        raise ValueError(f"Feature columns not found: {missing_features}")

    df_copy = df.copy()

    # Separate known and missing data
    known_mask = df_copy[target_col].notna()
    missing_mask = df_copy[target_col].isna()

    known = df_copy[known_mask]
    missing = df_copy[missing_mask]

    if len(missing) == 0:
        print(f"No missing values found in column '{target_col}'")
        return df

    if len(known) == 0:
        print(f"No known values found in column '{target_col}' for training")
        return df

    # Prepare training data
    X_train = known[feature_cols].copy()
    y_train = known[target_col].copy()
    X_test = missing[feature_cols].copy()

    # Check for missing values in features
    if X_train.isnull().any().any() or X_test.isnull().any().any():
        print(f"Warning: Missing values found in feature columns for '{target_col}'")
        # Fill missing features with median
        for col in feature_cols:
            median_val = X_train[col].median()
            X_train[col] = X_train[col].fillna(median_val)
            X_test[col] = X_test[col].fillna(median_val)

    try:
        # Train model
        model = HistGradientBoostingRegressor(random_state=42)
        model.fit(X_train, y_train)

        # Make predictions
        predicted_values = model.predict(X_test)

        if round_values:
            predicted_values = np.round(predicted_values).astype(int)

        # Fill missing values
        df_copy.loc[missing_mask, target_col] = predicted_values

        # Calculate and report R² score
        train_predictions = model.predict(X_train)
        r2 = r2_score(y_train, train_predictions)
        print(f"Model performance for '{target_col}': R² = {r2:.3f}")

        if r2 < 0.5:
            print(f"Warning: Low R² score ({r2:.3f}) for '{target_col}' - predictions may be unreliable")

    except Exception as e:
        print(f"Error training model for '{target_col}': {e}")
        return df

    return df_copy


def fill_missing_destinations_by_proximity(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill missing destination values based on geographic proximity within trips.

    Args:
        df: Input dataframe

    Returns:
        Dataframe with filled destination values
    """
    required_cols = ['TripID', 'Destination', 'Latitude', 'Longitude']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Required columns not found: {missing_cols}")

    df_filled = df.copy()
    total_filled = 0
    total_trips = df_filled['TripID'].nunique()

    print(f"Processing {total_trips} trips for destination proximity filling...")

    for trip_id, trip_group in df_filled.groupby('TripID'):
        # Ensure chronological order
        trip_group = trip_group.sort_values('time')
        trip_indices = trip_group.index

        trip_filled_count = 0

        for i, idx in enumerate(trip_indices):
            if pd.isna(df_filled.at[idx, 'Destination']) :
                current_lat = df_filled.at[idx, 'Latitude']
                current_lon = df_filled.at[idx, 'Longitude']

                # Skip if current position is invalid
                if pd.isna(current_lat) or pd.isna(current_lon):
                    print(f"Skipping index {idx} due to invalid coordinates")
                    continue

                # Find nearest known destination above current position
                above_dest = None
                for j in range(i - 1, -1, -1):
                    ref_idx = trip_indices[j]
                    if not pd.isna(df_filled.at[ref_idx, 'Destination']):
                        above_dest = {
                            'lat': df_filled.at[ref_idx, 'Latitude'],
                            'lon': df_filled.at[ref_idx, 'Longitude'],
                            'dest': df_filled.at[ref_idx, 'Destination']
                        }
                        break

                # Find nearest known destination below current position
                below_dest = None
                for j in range(i + 1, len(trip_indices)):
                    ref_idx = trip_indices[j]
                    if not pd.isna(df_filled.at[ref_idx, 'Destination']):
                        below_dest = {
                            'lat': df_filled.at[ref_idx, 'Latitude'],
                            'lon': df_filled.at[ref_idx, 'Longitude'],
                            'dest': df_filled.at[ref_idx, 'Destination']
                        }
                        break

                # Choose destination based on proximity
                chosen_dest = None
                if above_dest and below_dest:
                    # Calculate distances
                    current_point = (current_lat, current_lon)
                    above_point = (above_dest['lat'], above_dest['lon'])
                    below_point = (below_dest['lat'], below_dest['lon'])

                    try:
                        dist_above = euclidean(current_point, above_point)
                        dist_below = euclidean(current_point, below_point)

                        chosen_dest = above_dest['dest'] if dist_above <= dist_below else below_dest['dest']
                    except Exception:
                        # Fallback to above destination if distance calculation fails
                        chosen_dest = above_dest['dest']

                elif above_dest:
                    chosen_dest = above_dest['dest']
                elif below_dest:
                    chosen_dest = below_dest['dest']

                if chosen_dest:
                    df_filled.at[idx, 'Destination'] = chosen_dest
                    trip_filled_count += 1
                    total_filled += 1

    print(f"Filled {total_filled} missing destination values using proximity method")
    return df_filled


# Column mapping for renaming columns to standardized format
column_mapping = {
    'TripID': 'trip_id',
    'StartLatitude': 'start_latitude',
    'StartLongitude': 'start_longitude',
    'StartTime': 'start_time',
    'EndLatitude': 'end_latitude',
    'EndLongitude': 'end_longitude',
    'EndTime': 'end_time',
    'StartPort': 'start_port',
    'EndPort': 'end_port',
    'time': 'time_stamp',
    'shiptype': 'ship_type',
    'Length': 'length',
    'Breadth': 'breadth',
    'Draught': 'draught',
    'Latitude': 'latitude',
    'Longitude': 'longitude',
    'SOG': 'speed_over_ground',
    'COG': 'course_over_ground',
    'TH': 'true_heading',
    'Destination': 'destination',
}