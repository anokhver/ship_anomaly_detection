import re
from typing import List, Dict, Optional, Union

# ---------------- Destination filtering and matching ----------------
name_german = {
    'HAM': ["HAMBURG", "HAMBUG", "HH", "FINKENWERD", "FINKENWERDER", "BLEXEN", "ELBE"],
    'BRV': ["BREMERHAVEN", "BREMENHAVEN", "DEBHV", "BHV"],
    # 'HAM.FINKENWERDER': ["FINKENWERDER", "FINKENWERD"],
    # 'HAM.BLEXEN': ["BLEXEN"],
    # 'HAM.ELBE': ["ELBE"],
    'BRE': ["BREMEN"],
    'KEL': ["KIEL"],
    'STA': ["STADE", "STAD"],
    'BRB': ["BRUNSBUETTEL", "BRUNSBUETT"],
    'VTT': ["HIDDENSEE"],
    'WVN': ["WILHELMSHAVEN"],
    'country': ["DE"]
}

name_poland = {
    'GDN': ["GDANSK", "GDANK"],
    'GDY': ["GDYNIA", "GYDNIA", "GYDINIA", "GDYNA"],
    'SZZ': ["SZCZECIN", "PLSZCZECIN"],
    'country': ["PL"]
}

name_lithuania = {  # Fixed typo in variable name
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
    'KGD': ["KALININGRAD", "KAL"],  # Removed duplicate
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


def full_dict() -> List[Dict[str, List[str]]]:
    """Return list of all country port dictionaries."""
    return [
        name_german,
        name_poland,
        name_lithuania,  # Fixed typo
        name_sweden,
        name_russia,
        name_denmark,
        name_finland,
        name_belgium
    ]


def clean_destination(dest: Union[str, None]) -> Union[str, None]:
    """Clean and standardize destination strings by removing special characters and normalizing format.

    Args:
        dest: The destination string to clean

    Returns:
        Cleaned destination string in uppercase with standardized formatting, or None if input is invalid
    """
    if not isinstance(dest, str) or not dest.strip():
        return None

    dest = dest.upper().strip()

    # Remove all whitespace
    dest = re.sub(r'\s+', '', dest)

    # Keep only alphanumeric characters and specific symbols
    dest = re.sub(r'[^A-Z0-9./\\><-]', '', dest)

    # Normalize repeated symbols
    dest = re.sub(r'\.{2,}', '.', dest)
    dest = re.sub(r'\\{2,}', r'\\', dest)
    dest = re.sub(r'/{2,}', '/', dest)
    dest = re.sub(r'>{2,}', '>', dest)
    dest = re.sub(r'<{2,}', '<', dest)
    dest = re.sub(r'-{2,}', '-', dest)

    # Remove isolated dots (not between alphanumeric characters)
    dest = re.sub(r'(?<![A-Z0-9])\.(?![A-Z0-9])', '', dest)

    # Standardize separators to dots
    dest = dest.replace('/', '.').replace('\\', '.').replace('-', '.')

    # Remove leading/trailing dots and clean up multiple dots
    dest = re.sub(r'\.+', '.', dest).strip('.')

    return dest if dest else None


def match_names(name: Union[str, None],
                variants_dicts: Optional[List[Dict[str, List[str]]]] = None,
                test: bool = False) -> Optional[str]:
    """Match a port name against known variants and return the standardized key.

    Args:
        name: The port name to match. Can include variations or misspellings.
        variants_dicts: List of dictionaries containing port variants and country codes.
        test: If True, returns the original name if no match is found.
              If False, returns None when no match is found.

    Returns:
        The standardized key if a match is found, or the original name/None based on `test`.

    Example:
        >>> match_names("Hamburg")
        'DE.HAM'
        >>> match_names("GDANSK")
        'PL.GDN'
        >>> match_names("GDYNIAVIANOK")
        'PL.GDY'
    """
    if not isinstance(name, str) or not name.strip():
        return name if test else None

    if variants_dicts is None:
        variants_dicts = full_dict()

    for variant_dict in variants_dicts:
        country_codes = variant_dict.get('country', [''])
        country_code = country_codes[0]

        # Get all port keys (excluding 'country')
        port_keys = [k for k in variant_dict.keys() if k != 'country']

        for port_key in port_keys:
            # Check all variants for this port
            all_patterns = [port_key] + variant_dict[port_key]

            for pattern in all_patterns:
                if _matches_pattern(name, pattern, country_code):
                    return f"{country_code}.{port_key}"

    return name if test else None


def _matches_pattern(name: str, pattern: str, country_code: str) -> bool:
    """Check if a name matches a pattern, considering country code variations."""

    # Since names are already cleaned (uppercase, no spaces, normalized dots),
    # we can use simpler string operations

    # Exact matches (including country code variations)
    exact_matches = [
        pattern,
        f"{country_code}.{pattern}",
        f"{country_code}{pattern}",
        f"{pattern}.{country_code}",
        f"{pattern}{country_code}"
    ]

    if name in exact_matches:
        return True

    # Prefix/suffix matches (handles cases like "GDYNIAVIANOK" containing "GDYNIA")
    if name.startswith(pattern) or name.endswith(pattern):
        return True

    # Check if pattern appears with dot separators
    # Since names are already normalized with dots as separators
    dot_patterns = [
        f".{pattern}.",  # .PATTERN.
        f".{pattern}",  # .PATTERN (at end)
        f"{pattern}.",  # PATTERN. (at start)
    ]

    for dot_pattern in dot_patterns:
        if dot_pattern in name:
            return True

    # Check country code combinations with optional dots
    # Since separators are already normalized to dots
    country_combinations = [
        f"{country_code}.{pattern}",
        f"{country_code}{pattern}",
        f"{pattern}.{country_code}",
        f"{pattern}{country_code}",
        f".{country_code}.{pattern}",
        f".{country_code}{pattern}",
        f".{pattern}.{country_code}",
        f".{pattern}{country_code}"
    ]

    return any(combo in name for combo in country_combinations)


def get_all_ports() -> Dict[str, str]:
    """Get a dictionary of all known ports with their standardized codes.

    Returns:
        Dictionary mapping port names/variants to their standardized codes
    """
    all_ports = {}

    for variant_dict in full_dict():
        country_code = variant_dict.get('country', [''])[0]

        for port_key, variants in variant_dict.items():
            if port_key == 'country':
                continue

            standardized_code = f"{country_code}.{port_key}"

            # Add the port key itself
            all_ports[port_key] = standardized_code

            # Add all variants
            for variant in variants:
                all_ports[variant] = standardized_code

    return all_ports


def validate_port_data() -> bool:
    """Validate the port data structure for consistency.

    Returns:
        True if all port data is valid, False otherwise
    """
    try:
        for i, variant_dict in enumerate(full_dict()):
            # Check if country key exists
            if 'country' not in variant_dict:
                print(f"Warning: Dictionary {i} missing 'country' key")
                return False

            # Check if country has valid code
            country_codes = variant_dict['country']
            if not country_codes or not isinstance(country_codes, list):
                print(f"Warning: Dictionary {i} has invalid country codes")
                return False

            # Check port entries
            for key, variants in variant_dict.items():
                if key == 'country':
                    continue

                if not isinstance(variants, list):
                    print(f"Warning: Port {key} variants is not a list")
                    return False

                if not variants:
                    print(f"Warning: Port {key} has empty variants list")
                    return False

        return True
    except Exception as e:
        print(f"Error validating port data: {e}")
        return False


# Test function to verify the fix
def test_matching():
    """Test the matching functionality with various inputs."""
    test_cases = [
        "GDYNIAVIANOK",
        "GDYNIA",
        "HAMBURG",
        "HAMBURGPORT",
        "DE.HAM",
        "HAM.DE",
        "BREMEN.CITY",
        "PLGDYNIA"
    ]

    print("Testing port matching:")
    for test_case in test_cases:
        result = match_names(test_case)
        print(f"'{test_case}' -> {result}")


if __name__ == "__main__":
    test_matching()
