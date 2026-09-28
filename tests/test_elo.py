import pandas as pd

from src.elo import compute_elo_ratings, get_expected_score, update_ratings


def test_get_expected_score_equal():
    expected = get_expected_score(1500, 1500)
    assert expected == 0.5


def test_get_expected_score_different():
    # rating_a > rating_b, expected_a should be > 0.5
    expected_a = get_expected_score(1600, 1400)
    expected_b = get_expected_score(1400, 1600)
    assert expected_a > 0.75
    assert expected_a < 0.77
    assert abs(expected_a + expected_b - 1.0) < 1e-9


def test_update_ratings_win():
    # Player A wins, ratings are equal
    new_a, new_b = update_ratings(1500, 1500, score_a=1.0, k_factor=32)
    assert new_a == 1516.0
    assert new_b == 1484.0
    # Zero-sum check
    assert new_a + new_b == 3000.0


def test_update_ratings_loss():
    # Player A loses, ratings are equal
    new_a, new_b = update_ratings(1500, 1500, score_a=0.0, k_factor=32)
    assert new_a == 1484.0
    assert new_b == 1516.0
    # Zero-sum check
    assert new_a + new_b == 3000.0


def test_compute_elo_ratings_sequential():
    # Create a dummy DataFrame simulating matches:
    # 2026-01-01: A vs B, winner A (blue)
    # 2026-01-02: B vs C, winner B (blue)
    # 2026-01-03: C vs A, winner A (white)
    data = {
        "contest_id": [1, 2, 3],
        "date": ["2026-01-01", "2026-01-02", "2026-01-03"],
        "competition_name": [
            "World Championships 2026",
            "Grand Slam 2026",
            "Olympic Games 2026",
        ],
        "athlete_blue_id": ["A", "B", "C"],
        "athlete_white_id": ["B", "C", "A"],
        "winner_id": ["A", "B", "A"],
        "winner_color": ["blue", "blue", "white"],
    }
    df = pd.DataFrame(data)

    # Process
    df_elo, final_ratings = compute_elo_ratings(df, initial_elo=1500, base_k=32)

    # Assertions
    # Final ratings should be computed
    assert "A" in final_ratings
    assert "B" in final_ratings
    assert "C" in final_ratings

    # Check that pre-match ratings columns are created
    assert "blue_pre_elo" in df_elo.columns
    assert "white_pre_elo" in df_elo.columns
    assert "blue_post_elo" in df_elo.columns
    assert "white_post_elo" in df_elo.columns

    # Match 1: A vs B. Pre ratings should be 1500
    assert df_elo.loc[0, "blue_pre_elo"] == 1500
    assert df_elo.loc[0, "white_pre_elo"] == 1500

    # For Match 1, competition is "World Championships" -> weight 3 -> K = 96
    # A wins (score_a = 1.0) -> A gets 1500 + 96 * (1.0 - 0.5) = 1548
    # B gets 1452
    assert df_elo.loc[0, "blue_post_elo"] == 1548
    assert df_elo.loc[0, "white_post_elo"] == 1452

    # Match 2: B vs C. B has 1452, C has 1500
    assert df_elo.loc[1, "blue_pre_elo"] == 1452
    assert df_elo.loc[1, "white_pre_elo"] == 1500
