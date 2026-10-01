"""Spherical geometry for "what's near here": distances, bearings and R*Tree-friendly bounding boxes.

UI-free and numpy-only. The Earth is a sphere of mean radius 6371.0088 km (good to ~0.3%, which
is plenty for "what's within 2 km"). Latitudes are -90..90, longitudes -180..180.
"""
from __future__ import annotations

import math

import numpy as np

EARTH_RADIUS_KM = 6371.0088
COMPASS = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]


def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance in km; scalars or numpy arrays (broadcasting)."""
    p1, p2 = np.radians(lat1), np.radians(lat2)
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(np.radians(lon2 - lon1) / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def bearing_deg(lat1, lon1, lat2, lon2):
    """Initial compass bearing (0 = north, 90 = east) from point 1 to point 2, in [0, 360)."""
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dl = np.radians(lon2 - lon1)
    y = np.sin(dl) * np.cos(p2)
    x = np.cos(p1) * np.sin(p2) - np.sin(p1) * np.cos(p2) * np.cos(dl)
    return (np.degrees(np.arctan2(y, x)) + 360.0) % 360.0


def compass(bearing):
    """8-point compass name for a bearing in degrees (scalar or array)."""
    i = (np.floor((np.asarray(bearing) + 22.5) / 45.0) % 8).astype(int)
    return COMPASS[int(i)] if i.ndim == 0 else [COMPASS[k] for k in i]


def wrap_lon(lon: float) -> float:
    """Longitude normalised to [-180, 180)."""
    return (lon + 180.0) % 360.0 - 180.0


def bounding_boxes(lat: float, lon: float, radius_km: float) -> list[tuple[float, float, float, float]]:
    """Boxes (min_lat, max_lat, min_lon, max_lon) that together contain every point within radius_km.

    - A circle that reaches a pole spans all longitudes (and clips at the pole).
    - Otherwise the half-width in longitude is asin(sin(d) / cos(lat)), which widens towards the
      poles (a 100 km radius at 80 degrees north is ~5 degrees of longitude each way).
    - A circle crossing the antimeridian comes back as two boxes, one on each side of +-180.
    The boxes are slightly generous; callers filter with haversine_km afterwards.
    """
    d = min(radius_km / EARTH_RADIUS_KM, math.pi)           # angular radius, radians
    dlat = math.degrees(d)
    lo, hi = lat - dlat - 1e-9, lat + dlat + 1e-9
    if lo <= -90 or hi >= 90 or d >= math.pi / 2:
        return [(max(lo, -90.0), min(hi, 90.0), -180.0, 180.0)]
    dlon = math.degrees(math.asin(min(1.0, math.sin(d) / math.cos(math.radians(lat))))) + 1e-9
    west, east = lon - dlon, lon + dlon
    if dlon >= 180 or (west < -180 and east > 180):
        return [(lo, hi, -180.0, 180.0)]
    if west < -180:
        return [(lo, hi, west + 360.0, 180.0), (lo, hi, -180.0, east)]
    if east > 180:
        return [(lo, hi, west, 180.0), (lo, hi, -180.0, east - 360.0)]
    return [(lo, hi, west, east)]
