"""Shared constants: city bounding box, crime categories, severity weights, chart palette."""

CITY = "Chicago"

# Rough bounding box of the city. Coordinates outside it are treated as errors.
BBOX = {"lat_min": 41.60, "lat_max": 42.05, "lon_min": -87.95, "lon_max": -87.50}

VIOLENT_TYPES = {"HOMICIDE", "ROBBERY", "BATTERY", "ASSAULT"}
PROPERTY_TYPES = {
    "THEFT", "BURGLARY", "MOTOR VEHICLE THEFT", "CRIMINAL DAMAGE", "DECEPTIVE PRACTICE",
}

# Illustrative relative-harm weights used to rank hotspots by severity, NOT by raw count.
# These are assumptions for the demo, not an official index. Replace with your agency's
# own weighting (e.g. a Cambridge Crime Harm Index style scheme) if you have one.
SEVERITY = {
    "HOMICIDE": 10.0,
    "ROBBERY": 4.0,
    "ASSAULT": 3.0,
    "BATTERY": 3.0,
    "BURGLARY": 2.5,
    "MOTOR VEHICLE THEFT": 2.0,
    "CRIMINAL DAMAGE": 1.5,
    "NARCOTICS": 1.0,
    "THEFT": 1.0,
    "DECEPTIVE PRACTICE": 1.0,
}
DEFAULT_SEVERITY = 1.0

# Canonical socioeconomic indicator names used throughout the project.
SOCIO_COLS = [
    "poverty_pct",
    "unemployment_pct",
    "no_hs_diploma_pct",
    "crowded_housing_pct",
    "dependency_pct",
    "per_capita_income",
]
SOCIO_LABELS = {
    "poverty_pct": "Households below poverty (%)",
    "unemployment_pct": "Aged 16+ unemployed (%)",
    "no_hs_diploma_pct": "Aged 25+ without HS diploma (%)",
    "crowded_housing_pct": "Crowded housing (%)",
    "dependency_pct": "Aged <18 or >64 (%)",
    "per_capita_income": "Per-capita income ($)",
    "hardship_index": "Hardship index (1-100)",
}

# Chart palette: the validated reference palette (categorical slots 1-3, sequential blue
# ramp, blue<->red diverging pair, chrome tokens). Light surface only (PNG output).
PALETTE = {
    "surface": "#fcfcfb",
    "ink": "#0b0b0b",
    "ink2": "#52514e",
    "muted": "#898781",
    "grid": "#e1e0d9",
    "axis": "#c3c2b7",
    "blue": "#2a78d6",
    "orange": "#eb6834",
    "aqua": "#1baf7a",
    "red": "#e34948",
    "seq": ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#2a78d6", "#1c5cab", "#104281", "#0d366b"],
    "div_mid": "#f0efec",
}
