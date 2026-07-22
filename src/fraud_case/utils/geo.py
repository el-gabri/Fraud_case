"""Funcoes geograficas usadas na engenharia de features."""

from __future__ import annotations

import numpy as np


def haversine_distance(lat1, lon1, lat2, lon2):
    """Distancia Haversine (em km) entre dois pontos, vetorizada com numpy.

    Aceita escalares ou arrays/Series numpy-compatíveis.
    """
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    earth_radius_km = 6371
    return c * earth_radius_km
