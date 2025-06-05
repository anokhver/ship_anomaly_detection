import re
from fuzzywuzzy import fuzz, process

name_german = {
    'DEHAM': ["HAMBURG", "HAMBUG", "HH", "HAM"],
    'DEBRV': ["BREMERHAVEN", "BREMENHAVEN", "BRV", "DEBHV", "BHV"],
    'DEBRE': ["BREMEN", "BRE"],
    'DEKEL': ["KIEL", "KEL"],
    'DEHAM.FINKENWERDER': ["FINKENWERDER", "FINKENWERD"],
    'DEHAM.BLEXEN': ["BLEXEN"],
    'DESTA': ["STADE", "STAD", "STA"],
    'DEBRB': ["BRUNSBUETTEL", "BRUNSBUETT", "BRB"],
    'DEHAM.ELBE': ["ELBE"],
    'DEVTT': ["HIDDENSEE", "VTT"],
    'DEWVN': ["WILHELMSHAVEN", "WVN"],
    'country': ["DE"]
}

name_poland = {
    'PLGDN': ["GDANSK", "GDANK", "GDN"],
    'PLGDY': ["GDYNIA", "GYDNIA", "GYDINIA", "GDY", "GDYNA"],
    'SZCZECIN': ["SZCZECIN", "SZCZECIN", "SZZ"],
    'country': ["PL"]
}

name_lythuania = {
    'LTKLJ': ["KLAIPEDA", "KLJ"],
    'country': ["LT"]
}

name_sweden = {
    'SEHAD': ["HALMSTAD", "HAD"],
    'SENOK': ["NOK"],
    'SEAHU': ["AHUS", "AHU"],
    'country': ["SE"]
}

name_russia = {
    'RUKGD': ["KALININGRAD", "KALININGRAD", "KAL"],
    'country': ["RU"]
}

name_denmark = {
    'DKKOB': ["KOBENHAVN", "COPENHAGEN", "COPENHAGUE", "CPH"],
    'country': ["DK"]
}

name_finland = {
    'FIHKO': ["HANKO", "HKO"],
    'country': ["FI"]
}

name_belgium = {
    "BEANR": ["BEANR", "ANR"],
    'country': ["BE"]
}

full_dict = [
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

    dest = re.sub(r'[^A-Za-z0-9\s./\\><]', '', dest)  # Keep only alphanum + ./\\><
    dest = re.sub(r'\.{2,}', '.', dest)  # Reduce multiple dots to one
    dest = re.sub(r'[\\/]{2,}', '/', dest)  # Standardize slashes to single /
    dest = re.sub(r'>{2,}', '>', dest)  # Reduce multiple > to one
    dest = re.sub(r'<{2,}', '<', dest)  # Reduce multiple < to one
    dest = re.sub(r'(?<!\w)\.|\.(?!\w)', '', dest)  # Remove lonely dots
    dest = re.sub(r'\s+', '', dest).strip().upper()  # Normalize whitespace and uppercase

    return dest


def save_filtered_ports(df, country_name='Germany'):
    """Filter port data by country and save to CSV, keeping only relevant columns.

    Args:
        df (pd.DataFrame): DataFrame containing port data
        country_name (str): Country name to filter by (default: 'Germany')
    """
    # Filter rows where 'Main Port Name', 'Alternate Port Name', 'Country Code', or 'Region Name' contains "Hamburg" or "DE"
    filtered_ports = df[
        df['Country Code'].str.contains(country_name, case=False, na=False) |
        df['Region Name'].str.contains(country_name, case=False, na=False)
        ]

    columns_to_drop = [
        col for col in filtered_ports.columns
        if not any(keyword in col.lower() for keyword in (
            'port name',
            'region',
            'locode',
            'maximum vessel',
            'entrance width',
            'channel depth',
            'cargo pier depth',
        ))
    ]

    filtered_ports = filtered_ports.drop(columns=columns_to_drop)
    filtered_ports.to_csv(f'../data/{country_name}.csv', index=False)


def find_fuzzy_matches(destinations, threshold=80, scorer=fuzz.token_set_ratio, show_progress=False):
    """
    Find fuzzy matches among destination names.

    Parameters:
        destinations (list): List of destination strings to compare
        threshold (int): Minimum similarity score to consider a match (0-100)
        scorer: Fuzzy matching function (default: token_set_ratio)
        show_progress (bool): Whether to print progress during processing

    Returns:
        dict: Dictionary where keys are original names and values are lists of matches
              with their scores in format [(matched_name, score), ...]
    """
    matches = {}
    total = len(destinations)

    for i, dest in enumerate(destinations, 1):
        # Skip NAN/empty values
        if not dest or str(dest).strip().upper() in ('NAN', 'NULL', ''):
            continue

        if show_progress:
            print(f"Processing {i}/{total}: {dest[:30]}...", end='\r')

        # Find matches above threshold (excluding self)
        potential_matches = process.extract(
            dest,
            destinations,
            scorer=scorer,
            limit=None
        )

        # Filter matches
        good_matches = [
            (match, score)
            for match, score in potential_matches
            if score >= threshold and match != dest
        ]

        if good_matches:
            matches[dest] = good_matches

    if show_progress:
        print("\n" + "=" * 50)

    return matches


def print_fuzzy_matches(matches, min_score=0, group_similar=False):
    """
    Print fuzzy matching results in a readable format.

    Parameters:
        matches (dict): Output from find_fuzzy_matches
        min_score (int): Minimum score to display
        group_similar (bool): Whether to group similar matches together
    """
    if not matches:
        print("No matches found")
        return

    print(f"\nFuzzy matches (score ≥ {min_score}):")
    print("=" * 60)

    if group_similar:
        # Group similar matches to avoid duplicates
        already_matched = set()
        for dest in sorted(matches.keys()):
            if dest in already_matched:
                continue

            print(f"\nGroup: {dest}")
            print("-" * 50)

            # Include the original in the group
            all_in_group = {dest}

            for match, score in matches[dest]:
                if score >= min_score:
                    print(f"  → {match} (score: {score})")
                    all_in_group.add(match)

                    # Also include matches of matches
                    if match in matches:
                        for submatch, subscore in matches[match]:
                            if subscore >= min_score and submatch not in all_in_group:
                                print(f"    → {submatch} (score: {subscore})")
                                all_in_group.add(submatch)

            already_matched.update(all_in_group)
    else:
        # Simple listing
        for dest in sorted(matches.keys()):
            print(f"\n{dest} matches:")
            print("-" * 50)
            for match, score in matches[dest]:
                if score >= min_score:
                    print(f"  → {match} (score: {score})")


def match_names(name, variants_dicts):
    """
    Match a port name against known variants and return the standardized key.

    Args:
        name (str): The port name to match
        variants_dicts (list): List of dictionaries containing port variants,
            where each dictionary has standardized keys with variant spellings as values,
            and a 'country' key with country codes

    Returns:
        str: The matched standardized key if found, or original name if no match

    Example:
        >>> name_german = {
        ...     'DEHAM': ["HAMBURG", "HAMBUG", "HH", "HAM"],
        ...     'country': ["DE"]
        ... }
        >>> match_names("Hamburg", [name_german])
        'DEHAM'
    """
    if not isinstance(name, str):
        return name

    for variant_dict in variants_dicts:
        country_prefix = variant_dict.get('country', [None])[0]
        for key, variants in variant_dict.items():
            if key == 'country':
                continue

            for variant in variants:
                pattern = rf"({country_prefix}[\.\-_]?)?{variant}([\.\-_]?{country_prefix})?"
                match = re.search(pattern, name, re.IGNORECASE)
                if match:
                    return key
    return name
