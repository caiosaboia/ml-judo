import pandas as pd

from src.glicko2 import Glicko2Player, compute_glicko2_ratings, glicko2_expected_score


def test_glicko2_player_initialization():
    player = Glicko2Player()
    assert player.rating == 1500.0
    assert player.rd == 350.0
    assert abs(player.volatility - 0.06) < 1e-5

def test_glicko2_expected_score():
    p1 = Glicko2Player(rating=1500.0, rd=350.0)
    p2 = Glicko2Player(rating=1500.0, rd=350.0)
    exp = glicko2_expected_score(p1, p2)
    assert abs(exp - 0.5) < 1e-4

def test_glicko2_update_winner_rd_reduction():
    p1 = Glicko2Player(rating=1500.0, rd=350.0)
    p2 = Glicko2Player(rating=1500.0, rd=350.0)
    
    new_p1, new_p2 = p1.update_against(p2, score=1.0)
    
    # Winner's rating should increase
    assert new_p1.rating > 1500.0
    # Loser's rating should decrease
    assert new_p2.rating < 1500.0
    # Both players' RD should decrease after a match
    assert new_p1.rd < 350.0
    assert new_p2.rd < 350.0

def test_compute_glicko2_ratings_sequential():
    data = {
        'contest_id': [1, 2],
        'date': ['2026-01-01', '2026-01-02'],
        'competition_name': ['Grand Slam 2026', 'World Championships 2026'],
        'athlete_blue_id': ['A', 'B'],
        'athlete_white_id': ['B', 'C'],
        'winner_id': ['A', 'B'],
        'winner_color': ['blue', 'blue']
    }
    df = pd.DataFrame(data)
    
    df_glicko, final_players = compute_glicko2_ratings(df)
    
    assert 'A' in final_players
    assert 'B' in final_players
    assert 'C' in final_players
    
    assert 'blue_pre_glicko' in df_glicko.columns
    assert 'white_pre_glicko' in df_glicko.columns
    assert df_glicko.loc[0, 'blue_pre_glicko'] == 1500.0
    assert df_glicko.loc[0, 'white_pre_glicko'] == 1500.0
