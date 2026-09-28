import pandas as pd


def classify_competition(name: str) -> tuple[int, str]:
    """
    Classifies a judo competition and assigns it an importance weight (Elo multiplier).

    Weights:
      3: Jogos Olímpicos, Campeonato Mundial, World Masters
      2: Grand Slam, Grand Prix
      1: Campeonatos Continentais, Copas Continentais e Nacionais (European Open, World Cup, Games, etc.)
    """
    if not isinstance(name, str):
        return 1, "Copas Continentais e Nacionais"
    name_lower = name.lower()

    # Weight 3
    if "olympic" in name_lower:
        return 3, "Jogos Olímpicos"
    if "world" in name_lower and "championship" in name_lower:
        return 3, "Campeonato Mundial"
    if "masters" in name_lower:
        return 3, "World Masters"

    # Weight 2
    if "grand slam" in name_lower:
        return 2, "Grand Slam"
    if "grand prix" in name_lower:
        return 2, "Grand Prix"

    # Weight 1
    if "championship" in name_lower:
        return 1, "Campeonatos Continentais"

    return 1, "Copas Continentais e Nacionais"


def load_and_clean_data(csv_path: str) -> pd.DataFrame:
    """
    Loads, cleans, and sorts the judo match dataset.
    """
    # Load dataset
    df = pd.read_csv(csv_path, low_memory=False)

    # Ensure contest_id is numeric and sorted correctly
    df["contest_id"] = pd.to_numeric(df["contest_id"], errors="coerce")
    df = df.dropna(subset=["contest_id"])
    df["contest_id"] = df["contest_id"].astype(int)

    # Convert date to datetime for chronological sorting
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])

    # Drop rows without a valid winner
    df = df.dropna(subset=["winner_id", "winner_color"])
    df["winner_id"] = pd.to_numeric(df["winner_id"], errors="coerce")
    df = df.dropna(subset=["winner_id"])
    df["winner_id"] = df["winner_id"].astype(int)

    # Sort chronologically by date and contest_id
    df = df.sort_values(by=["date", "contest_id"]).reset_index(drop=True)

    # Add weight and category columns
    weights_and_categories = df["competition_name"].apply(classify_competition)
    df["competition_weight"] = [wc[0] for wc in weights_and_categories]
    df["competition_category"] = [wc[1] for wc in weights_and_categories]

    return df
