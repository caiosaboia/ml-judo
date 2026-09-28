import pandas as pd

from src.advanced_models import train_lightgbm_model, train_victory_method_model
from src.elo_advanced import compute_margin_weighted_elo


def test_margin_weighted_elo():
    data = {
        "contest_id": [1, 2],
        "date": ["2026-01-01", "2026-01-02"],
        "competition_name": ["Grand Slam 2026", "Grand Prix 2026"],
        "athlete_blue_id": ["A", "B"],
        "athlete_white_id": ["B", "C"],
        "winner_id": ["A", "B"],
        "winner_color": ["blue", "blue"],
        "blue_ippon": [1, 0],
        "white_ippon": [0, 0],
        "blue_wazari": [0, 1],
        "white_wazari": [0, 0],
        "blue_shido": [0, 0],
        "white_shido": [0, 2],
    }
    df = pd.DataFrame(data)
    df_result, ratings = compute_margin_weighted_elo(df)

    assert "blue_pre_margin_elo" in df_result.columns
    assert "A" in ratings
    # Ippon win should grant a higher post elo than a decision/wazari win under equal competition weight
    assert df_result.loc[0, "blue_post_margin_elo"] > 1500.0


def test_lightgbm_pipeline():
    X = pd.DataFrame(
        {
            "elo_diff": [
                100.0,
                -50.0,
                200.0,
                -100.0,
                50.0,
                0.0,
                -20.0,
                80.0,
                120.0,
                -80.0,
            ],
            "blue_pre_elo": [
                1600,
                1450,
                1700,
                1400,
                1550,
                1500,
                1480,
                1580,
                1620,
                1420,
            ],
            "white_pre_elo": [
                1500,
                1500,
                1500,
                1500,
                1500,
                1500,
                1500,
                1500,
                1500,
                1500,
            ],
            "height_diff": [5.0, -2.0, 10.0, -5.0, 0.0, 2.0, -1.0, 3.0, 4.0, -3.0],
            "weight_diff": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        }
    )
    y = pd.Series([1, 0, 1, 0, 1, 0, 0, 1, 1, 0])

    results = train_lightgbm_model(X, y, split_ratio=0.7)
    assert "model" in results
    assert "metrics" in results
    assert results["metrics"]["accuracy"] >= 0.0


def test_victory_method_model():
    X = pd.DataFrame(
        {
            "elo_diff": [100.0, -50.0, 200.0, -100.0, 50.0, 0.0],
            "blue_pre_elo": [1600, 1450, 1700, 1400, 1550, 1500],
            "white_pre_elo": [1500, 1500, 1500, 1500, 1500, 1500],
        }
    )
    y_method = pd.Series(["ippon", "wazari", "ippon", "shido", "wazari", "shido"])

    results = train_victory_method_model(X, y_method, split_ratio=0.6)
    assert "model" in results
    assert "accuracy" in results["metrics"]
