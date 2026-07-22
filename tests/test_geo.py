import numpy as np

from fraud_case.utils.geo import haversine_distance


def test_distance_to_self_is_zero():
    assert haversine_distance(10.0, 20.0, 10.0, 20.0) == 0.0


def test_known_distance_new_york_los_angeles():
    # Distancia real NY-LA e de aproximadamente 3936 km.
    ny = (40.7128, -74.0060)
    la = (34.0522, -118.2437)
    distance = haversine_distance(ny[0], ny[1], la[0], la[1])
    assert 3900 <= distance <= 3970


def test_vectorized_matches_scalar():
    lat1 = np.array([10.0, 20.0, -5.0])
    lon1 = np.array([20.0, 30.0, 10.0])
    lat2 = np.array([11.0, 20.0, -6.0])
    lon2 = np.array([21.0, 30.0, 11.0])

    vector_result = haversine_distance(lat1, lon1, lat2, lon2)
    for i in range(len(lat1)):
        scalar_result = haversine_distance(lat1[i], lon1[i], lat2[i], lon2[i])
        assert np.isclose(vector_result[i], scalar_result)
