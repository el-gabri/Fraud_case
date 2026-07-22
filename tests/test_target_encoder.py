import pandas as pd

from fraud_case.features.pipeline import SmoothedTargetEncoder


def test_unseen_category_falls_back_to_global_mean():
    X = pd.DataFrame({"job": ["a", "a", "b", "b", "a", "a", "a", "a", "a", "a"]})
    y = pd.Series([1, 1, 0, 0, 1, 1, 1, 1, 1, 1])

    encoder = SmoothedTargetEncoder(columns=["job"], smoothing=20, min_samples_leaf=10)
    encoder.fit(X, y)

    X_test = pd.DataFrame({"job": ["never_seen_before"]})
    encoded = encoder.transform(X_test)

    assert encoded["job_target_enc"].iloc[0] == encoder.global_mean_


def test_high_count_category_pulls_toward_its_own_mean():
    # "b" tem poucas amostras (deve ficar perto da media global);
    # "a" tem muitas amostras com media bem diferente da global (deve
    # ficar mais perto da propria media do que da global).
    n_a = 200
    X = pd.DataFrame({"job": ["a"] * n_a + ["b", "b"]})
    y = pd.Series([1] * n_a + [0, 0])  # media(a)=1.0, media(b)=0.0, media global ~= 0.99

    encoder = SmoothedTargetEncoder(columns=["job"], smoothing=20, min_samples_leaf=10)
    encoder.fit(X, y)
    encoded = encoder.transform(X)

    encoded_a = encoded.loc[X["job"] == "a", "job_target_enc"].iloc[0]
    encoded_b = encoded.loc[X["job"] == "b", "job_target_enc"].iloc[0]

    global_mean = encoder.global_mean_
    assert abs(encoded_a - 1.0) < abs(global_mean - 1.0)  # "a" mais perto da propria media
    assert abs(encoded_b - global_mean) < abs(0.0 - global_mean)  # "b" puxado para a media global


def test_get_feature_names_out():
    encoder = SmoothedTargetEncoder(columns=["job", "city"])
    names = encoder.get_feature_names_out()
    assert list(names) == ["job_target_enc", "city_target_enc"]
