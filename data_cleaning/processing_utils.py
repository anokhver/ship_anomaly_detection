import re

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.spatial import distance
import seaborn as sns


def plot_missing(df):
    cols = df.columns
    plt.figure(figsize=(10, 5))
    return sns.heatmap(df[cols].isnull(), cmap=['white', 'black'], cbar=False)


def get_percentage_missing(df, col=None):
    if col is None:
        return df.isnull().sum() / len(df) * 100
    else:
        return df[col].isnull().sum() / len(df) * 100


def get_entries_with_missing_values(df, col):
    trips_with_missing = df[df[col].isna()][
        'TripID'].unique()  # 1. Get TripIDs with at least one missing col value
    df_missing_trips = df[df['TripID'].isin(trips_with_missing)]  # 2. Filter DataFrame to include only these trips

    return (
        df_missing_trips
        .groupby('TripID')[col]
        .apply(lambda x: list(x.unique()))
    )


# ---------------- Destination filtering and matching ----------------
name_german = {
    'HAM': ["HAMBURG", "HAMBUG", "HH", "FINKENWERD", "FINKENWERDER", "BLEXEN", "ELBE"],
    # 'HAM.FINKENWERDER': ["FINKENWERDER", "FINKENWERD"],
    # 'DEHAM.BLEXEN': ["BLEXEN"],
    # 'DEHAM.ELBE': ["ELBE"],
    'BRV': ["BREMERHAVEN", "BREMENHAVEN", "DEBHV", "BHV"],
    'BRE': ["BREMEN"],
    'KEL': ["KIEL"],
    'STA': ["STADE", "STAD"],
    'BRB': ["BRUNSBUETTEL", "BRUNSBUETT"],
    'VTT': ["HIDDENSEE"],
    'WVN': ["WILHELMSHAVEN", ],
    'country': ["DE"]
}

name_poland = {
    'GDN': ["GDANSK", "GDANK"],
    'GDY': ["GDYNIA", "GYDNIA", "GYDINIA", "GDYNA"],
    'SZZ': ["SZCZECIN", "PLSZCZECIN"],
    'country': ["PL"]
}

name_lythuania = {
    'KLJ': ["KLAIPEDA"],
    'country': ["LT"]
}

name_sweden = {
    'HAD': ["HALMSTAD"],
    'NOK': ["NOK"],
    'AHU': ["AHUS"],
    'country': ["SE"]
}

name_russia = {
    'KGD': ["KALININGRAD", "KALININGRAD", "KAL"],
    'country': ["RU"]
}

name_denmark = {
    'KOB': ["KOBENHAVN", "COPENHAGEN", "COPENHAGUE", "CPH"],
    'country': ["DK"]
}

name_finland = {
    'HKO': ["HANKO"],
    'country': ["FI"]
}

name_belgium = {
    "ANR": ["ANR"],
    'country': ["BE"]
}


def full_dict():
    return [
        name_german,
        name_poland,
        name_lythuania,
        name_sweden,
        name_russia,
        name_denmark,
        name_finland,
        name_belgium
    ]


def clean_destination(dest):
    """Clean and standardize destination strings by removing special characters and normalizing format.

    Args:
        dest (str): The destination string to clean

    Returns:
        str: Cleaned destination string in uppercase with standardized formatting
    """
    if not isinstance(dest, str):
        return dest  # Skip non-string values

    dest = dest.upper()
    dest = re.sub(r'\s+', '', dest)  # Remove all whitespace

    dest = re.sub(r'[^A-Za-z0-9./\\><]', '', dest)  # Keep only alphanum + ./\\><

    # Normalize slashes and symbols
    dest = re.sub(r'\.{2,}', '.', dest)
    dest = re.sub(r'\\{2,}', r'\\', dest)
    dest = re.sub(r'/{2,}', '/', dest)
    dest = re.sub(r'>{2,}', '>', dest)
    dest = re.sub(r'<{2,}', '<', dest)
    dest = re.sub(r'(?<!\w)\.|\.(?!\w)', '', dest)  # Remove lonely dots

    # Replace slashes with dots
    dest = dest.replace('/', '.').replace('\\', '.')

    # Remove leading/trailing dots
    dest = dest.strip('.')

    return dest


def match_names(name, variants_dicts=None, test=False):
    """
    Match a port name against known variants and return the standardized key.

    Args:
        name (str): The port name to match. Can include variations or misspellings.
        variants_dicts (list): A list of dictionaries where:
            - Keys are standardized port codes or names.
            - Values are lists of variant spellings or aliases for the key.
            - Each dictionary also contains a 'country' key with a list of country codes.
        test (bool): If True, returns the original name if no match is found.
                     If False, returns None when no match is found.

    Returns:
        str: The standardized key if a match is found, or the original name/None based on `test`.

    Example:
        >>> name_german = {
        ...     'DEHAM': ["HAMBURG", "HAMBUG", "HH", "HAM"],
        ...     'country': ["DE"]
        ... }
        >>> match_names("Hamburg", [name_german])
        'DEHAM'

    Notes:
        - The function uses regex to match names, accounting for optional country prefixes or suffixes.
        - Matching prioritizes longer keys to avoid partial matches.
    """
    if not isinstance(name, str, ):
        return name if test else None
    if variants_dicts is None:
        variants_dicts = full_dict()

    for variant_dict in variants_dicts:
        country_prefix = variant_dict.get('country', [None])[0]

        # Get all keys sorted by specificity (longest first)
        sorted_keys = sorted(
            [k for k in variant_dict.keys() if k != 'country'],
            key=lambda x: -len(x)
        )

        for key in sorted_keys:
            # Try matching both the key itself and its variants
            patterns_to_try = [key] + variant_dict[key]

            for pattern_str in patterns_to_try:
                # Build regex pattern that accounts for country prefix/suffix
                pattern = rf"({country_prefix}[\.\-_]?)?{pattern_str}([\.\-_]?{country_prefix})?"
                if re.search(pattern, name, re.IGNORECASE):
                    return f"{country_prefix}.{key}"
                    # before = name[:match.start()].strip()
                    # after = name[match.end():].strip()
                    #
                    # replacement = (
                    #     f"{before + '.' if before and re.search(r'[A-Za-z]$', before) else before or ''}"
                    #     f"{key}"
                    #     f"{'.' + after if after and re.search(r'^[A-Za-z]', after) else after or ''}"
                    # )
    return name if test else None


def fill_missing_destinations_by_proximity(df):
    df_filled = df.copy()

    for trip_id, trip_group in df_filled.groupby('TripID'):
        # Ensure order
        trip_group = trip_group.sort_index()
        trip_indices = trip_group.index

        for i, idx in enumerate(trip_indices):
            if pd.isna(df_filled.at[idx, 'Destination']):
                above = None
                for j in range(i - 1, -1, -1):
                    ref_idx = trip_indices[j]
                    if not pd.isna(df_filled.at[ref_idx, 'Destination']):
                        above = {
                            'index': ref_idx,
                            'lat': df_filled.at[ref_idx, 'Latitude'],
                            'lon': df_filled.at[ref_idx, 'Longitude'],
                            'dest': df_filled.at[ref_idx, 'Destination']
                        }
                        break

                below = None
                for j in range(i + 1, len(trip_indices)):
                    ref_idx = trip_indices[j]
                    if not pd.isna(df_filled.at[ref_idx, 'Destination']):
                        below = {
                            'index': ref_idx,
                            'lat': df_filled.at[ref_idx, 'Latitude'],
                            'lon': df_filled.at[ref_idx, 'Longitude'],
                            'dest': df_filled.at[ref_idx, 'Destination']
                        }
                        break

                if above and below:
                    current_point = np.array([[df_filled.at[idx, 'Latitude'],
                                               df_filled.at[idx, 'Longitude']]])
                    above_point = np.array([[above['lat'], above['lon']]])
                    below_point = np.array([[below['lat'], below['lon']]])

                    dist_above = distance.cdist(current_point, above_point)[0][0]
                    dist_below = distance.cdist(current_point, below_point)[0][0]

                    df_filled.at[idx, 'Destination'] = (
                        above['dest'] if dist_above <= dist_below else below['dest']
                    )
                elif above:
                    df_filled.at[idx, 'Destination'] = above['dest']
                elif below:
                    df_filled.at[idx, 'Destination'] = below['dest']

    return df_filled
