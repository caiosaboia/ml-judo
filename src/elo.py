import pandas as pd


def get_expected_score(rating_a: float, rating_b: float, scale: float = 400.0) -> float:
    """
    Calculates the expected score of player A against player B using the Elo formula.
    """
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / scale))


def update_ratings(
    rating_a: float,
    rating_b: float,
    score_a: float,
    k_factor: float,
    scale: float = 400.0,
) -> tuple[float, float]:
    """
    Updates the ratings of player A and B based on the match score and K-factor.
    """
    expected_a = get_expected_score(rating_a, rating_b, scale)
    expected_b = 1.0 - expected_a

    score_b = 1.0 - score_a

    new_rating_a = rating_a + k_factor * (score_a - expected_a)
    new_rating_b = rating_b + k_factor * (score_b - expected_b)

    return new_rating_a, new_rating_b


def compute_elo_ratings(
    df: pd.DataFrame,
    initial_elo: float = 1500.0,
    base_k: float = 32.0,
    scale: float = 400.0,
) -> tuple[pd.DataFrame, dict[int, float]]:
    """
    Computes Elo ratings chronologically for all matches in the DataFrame.

    Assumes df is already sorted by date and contest_id.
    """
    ratings = {}

    blue_pre_elos = []
    white_pre_elos = []
    blue_post_elos = []
    white_post_elos = []

    for row in df.itertuples():
        b_id = row.athlete_blue_id
        w_id = row.athlete_white_id

        # Get current ratings or default to initial_elo
        b_rating = ratings.get(b_id, initial_elo)
        w_rating = ratings.get(w_id, initial_elo)

        blue_pre_elos.append(b_rating)
        white_pre_elos.append(w_rating)

        # Determine actual scores (1 = win, 0 = loss)
        # winner_color is either 'blue' or 'white'
        if row.winner_color == "blue":
            score_blue = 1.0
        else:
            score_blue = 0.0

        # K-factor = base_k * competition_weight
        if "competition_weight" in df.columns:
            weight = row.competition_weight
        elif "competition_name" in df.columns:
            from src.data_processing import classify_competition

            weight, _ = classify_competition(row.competition_name)
        else:
            weight = 1

        k = base_k * weight

        # Compute new ratings
        new_b_rating, new_w_rating = update_ratings(
            b_rating, w_rating, score_blue, k, scale
        )

        blue_post_elos.append(new_b_rating)
        white_post_elos.append(new_w_rating)

        # Store updated ratings
        ratings[b_id] = new_b_rating
        ratings[w_id] = new_w_rating

    df_result = df.copy()
    df_result["blue_pre_elo"] = blue_pre_elos
    df_result["white_pre_elo"] = white_pre_elos
    df_result["blue_post_elo"] = blue_post_elos
    df_result["white_post_elo"] = white_post_elos

    return df_result, ratings
