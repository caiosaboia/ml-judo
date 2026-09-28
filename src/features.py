import numpy as np
import pandas as pd


def preprocess_and_engineer_features(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Cleans heights/weights, imputes missing values using category averages,
    engineers difference features, encodes categoricals, and returns X and y.
    """
    df = df.copy()

    # 1. Clean zeros to NaN for imputation
    df["blue_height"] = df["blue_height"].replace(0.0, np.nan)
    df["white_height"] = df["white_height"].replace(0.0, np.nan)
    df["blue_weight"] = df["blue_weight"].replace(0.0, np.nan)
    df["white_weight"] = df["white_weight"].replace(0.0, np.nan)

    # Ensure weight_category is numeric
    df["weight_category"] = pd.to_numeric(df["weight_category"], errors="coerce")
    # If NaN in weight_category, fill with blue_weight or white_weight or default
    df["weight_category"] = (
        df["weight_category"]
        .fillna(df["blue_weight"])
        .fillna(df["white_weight"])
        .fillna(0)
    )

    # 2. Impute missing heights/weights using gender & weight category averages
    # Combine blue and white data to get overall statistics
    blue_stats = df[
        ["blue_gender", "weight_category", "blue_height", "blue_weight"]
    ].rename(
        columns={
            "blue_gender": "gender",
            "blue_height": "height",
            "blue_weight": "weight",
        }
    )
    white_stats = df[
        ["white_gender", "weight_category", "white_height", "white_weight"]
    ].rename(
        columns={
            "white_gender": "gender",
            "white_height": "height",
            "white_weight": "weight",
        }
    )
    combined_stats = pd.concat([blue_stats, white_stats], ignore_index=True)

    # Compute group averages
    group_means = (
        combined_stats.groupby(["gender", "weight_category"]).mean().reset_index()
    )
    overall_height = combined_stats["height"].mean()
    overall_weight = combined_stats["weight"].mean()

    # Standard fallback if empty
    if pd.isna(overall_height):
        overall_height = 175.0  # fallback standard height in cm
    if pd.isna(overall_weight):
        overall_weight = 73.0  # fallback standard weight in kg

    # Impute Blue
    df = df.merge(
        group_means.rename(
            columns={"height": "blue_mean_height", "weight": "blue_mean_weight"}
        ),
        left_on=["blue_gender", "weight_category"],
        right_on=["gender", "weight_category"],
        how="left",
    )
    df.drop(columns=["gender"], errors="ignore", inplace=True)
    df["blue_height"] = (
        df["blue_height"].fillna(df["blue_mean_height"]).fillna(overall_height)
    )
    df["blue_weight"] = (
        df["blue_weight"].fillna(df["blue_mean_weight"]).fillna(overall_weight)
    )
    df.drop(
        columns=["blue_mean_height", "blue_mean_weight"], errors="ignore", inplace=True
    )

    # Impute White
    df = df.merge(
        group_means.rename(
            columns={"height": "white_mean_height", "weight": "white_mean_weight"}
        ),
        left_on=["white_gender", "weight_category"],
        right_on=["gender", "weight_category"],
        how="left",
    )
    df.drop(columns=["gender"], errors="ignore", inplace=True)
    df["white_height"] = (
        df["white_height"].fillna(df["white_mean_height"]).fillna(overall_height)
    )
    df["white_weight"] = (
        df["white_weight"].fillna(df["white_mean_weight"]).fillna(overall_weight)
    )
    df.drop(
        columns=["white_mean_height", "white_mean_weight"],
        errors="ignore",
        inplace=True,
    )

    # 3. Engineer features
    df["elo_diff"] = df["blue_pre_elo"] - df["white_pre_elo"]
    df["height_diff"] = df["blue_height"] - df["white_height"]
    df["weight_diff"] = df["blue_weight"] - df["white_weight"]

    # Target: 1 if blue wins, 0 if white wins
    y = (df["winner_color"] == "blue").astype(int)

    # Features selection
    features_num = [
        "blue_pre_elo",
        "white_pre_elo",
        "elo_diff",
        "blue_height",
        "white_height",
        "height_diff",
        "blue_weight",
        "white_weight",
        "weight_diff",
        "weight_category",
    ]

    X = df[features_num].copy()

    # Categorical: Gender
    X["is_male"] = (df["blue_gender"] == "M").astype(int)

    # Categorical: Country
    # To keep dimensionality low and avoid overfitting, we identify the top 15 countries
    # and map the rest to "OTHER"
    top_countries = df["blue_country"].value_counts().index[:15].tolist()

    blue_country_clean = df["blue_country"].apply(
        lambda x: x if x in top_countries else "OTHER"
    )
    white_country_clean = df["white_country"].apply(
        lambda x: x if x in top_countries else "OTHER"
    )

    # One-hot encode countries
    blue_country_dummies = pd.get_dummies(
        blue_country_clean, prefix="blue_country", dtype=int
    )
    white_country_dummies = pd.get_dummies(
        white_country_clean, prefix="white_country", dtype=int
    )

    X = pd.concat([X, blue_country_dummies, white_country_dummies], axis=1)

    return X, y
