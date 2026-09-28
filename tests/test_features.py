import pandas as pd

from src.features import preprocess_and_engineer_features


def test_preprocess_and_engineer_features():
    # Setup dummy data with missing heights and weights
    data = {
        'blue_pre_elo': [1600, 1500],
        'white_pre_elo': [1500, 1400],
        'blue_height': [180.0, 0.0],  # 0.0 should be imputed
        'white_height': [0.0, 170.0], # 0.0 should be imputed
        'blue_weight': [81.0, 73.0],
        'white_weight': [81.0, 0.0],  # 0.0 should be imputed
        'blue_gender': ['M', 'F'],
        'white_gender': ['M', 'F'],
        'weight_category': [-81, -73],
        'blue_country': ['BRA', 'FRA'],
        'white_country': ['FRA', 'JPN'],
        'winner_color': ['blue', 'white'],
    }
    df = pd.DataFrame(data)
    
    # Process
    X, y = preprocess_and_engineer_features(df)
    
    # Assert no missing values in the output features
    assert X.isna().sum().sum() == 0
    
    # Assert elo_diff calculations
    assert X.loc[0, 'elo_diff'] == 100
    assert X.loc[1, 'elo_diff'] == 100
    
    # Check that heights were imputed (no 0.0 height should remain)
    assert X.loc[1, 'blue_height'] > 0.0
    assert X.loc[0, 'white_height'] > 0.0
    
    # Assert correct shapes
    assert len(X) == 2
    assert len(y) == 2
    assert y.tolist() == [1, 0] # blue wins = 1, white wins = 0
