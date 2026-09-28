import pandas as pd

from src.elo import update_ratings


def get_margin_multiplier(row: pd.Series | tuple) -> float:
    """
    Returns a margin multiplier based on how the match was won:
    - Ippon (decisive throw/pin/submission): 1.5x
    - Wazari (technical score): 1.2x
    - Decision / Penalty (Shido/Hansoku-make): 1.0x
    """
    b_ippon = getattr(row, 'blue_ippon', 0)
    w_ippon = getattr(row, 'white_ippon', 0)
    b_wazari = getattr(row, 'blue_wazari', 0)
    w_wazari = getattr(row, 'white_wazari', 0)

    try:
        b_ippon = int(b_ippon) if pd.notna(b_ippon) else 0
        w_ippon = int(w_ippon) if pd.notna(w_ippon) else 0
        b_wazari = int(b_wazari) if pd.notna(b_wazari) else 0
        w_wazari = int(w_wazari) if pd.notna(w_wazari) else 0
    except (ValueError, TypeError):
        b_ippon, w_ippon, b_wazari, w_wazari = 0, 0, 0, 0

    if b_ippon > 0 or w_ippon > 0:
        return 1.5
    elif b_wazari > 0 or w_wazari > 0:
        return 1.2
    return 1.0

def compute_margin_weighted_elo(df: pd.DataFrame, initial_elo: float = 1500.0, base_k: float = 32.0, scale: float = 400.0) -> tuple[pd.DataFrame, dict[int, float]]:
    """
    Computes Elo ratings chronologically using victory-margin K-factor multipliers.
    """
    ratings = {}

    blue_pre_elos = []
    white_pre_elos = []
    blue_post_elos = []
    white_post_elos = []

    for row in df.itertuples():
        b_id = row.athlete_blue_id
        w_id = row.athlete_white_id

        b_rating = ratings.get(b_id, initial_elo)
        w_rating = ratings.get(w_id, initial_elo)

        blue_pre_elos.append(b_rating)
        white_pre_elos.append(w_rating)

        score_blue = 1.0 if row.winner_color == 'blue' else 0.0

        comp_weight = getattr(row, 'competition_weight', 1)
        margin_mult = get_margin_multiplier(row)
        
        effective_k = base_k * comp_weight * margin_mult

        new_b, new_w = update_ratings(b_rating, w_rating, score_blue, effective_k, scale)

        blue_post_elos.append(new_b)
        white_post_elos.append(new_w)

        ratings[b_id] = new_b
        ratings[w_id] = new_w

    df_res = df.copy()
    df_res['blue_pre_margin_elo'] = blue_pre_elos
    df_res['white_pre_margin_elo'] = white_pre_elos
    df_res['blue_post_margin_elo'] = blue_post_elos
    df_res['white_post_margin_elo'] = white_post_elos

    return df_res, ratings
