import os
import sys

from src.advanced_models import (
    parse_victory_method_target,
    train_lightgbm_model,
    train_victory_method_model,
)
from src.config import BASE_K, CSV_PATH, ELO_SCALE, INITIAL_ELO, PLOTS_DIR
from src.data_processing import load_and_clean_data
from src.elo import compute_elo_ratings
from src.elo_advanced import compute_margin_weighted_elo
from src.features import preprocess_and_engineer_features
from src.glicko2 import compute_glicko2_ratings
from src.model import train_and_evaluate_model
from src.visualization import generate_and_save_plots


def main():
    print("==================================================")
    print("   JUDO ELO & MACHINE LEARNING ANALYTICS PIPELINE ")
    print("==================================================")
    
    # 1. Load and clean the CSV dataset
    print(f"\n[1/6] Loading dataset from: {CSV_PATH}")
    if not os.path.exists(CSV_PATH):
        print(f"Error: Dataset file not found at {CSV_PATH}!")
        sys.exit(1)
        
    df_clean = load_and_clean_data(CSV_PATH)
    print(f"Dataset loaded. Total clean matches for analysis: {len(df_clean)}")
    
    # 2. Compute Elo, Glicko-2, and Margin Elo ratings
    print(f"\n[2/6] Computing Elo (Base K={BASE_K}), Glicko-2 (RD=350), and Margin-Weighted Elo...")
    df_elo, final_ratings = compute_elo_ratings(df_clean, initial_elo=INITIAL_ELO, base_k=BASE_K, scale=ELO_SCALE)
    _, _ = compute_glicko2_ratings(df_clean)
    _, _ = compute_margin_weighted_elo(df_clean)
    print(f"Ratings complete. Rated athletes: {len(final_ratings)}")
    
    # 3. Preprocess and engineer features
    print("\n[3/6] Engineering features & imputing physical attributes...")
    X, y = preprocess_and_engineer_features(df_elo)
    print(f"Features ready. Shape: X={X.shape}, y={y.shape}")
    
    # 4. Train Random Forest & LightGBM Models
    print("\n[4/6] Training Random Forest and LightGBM Classifiers...")
    rf_results = train_and_evaluate_model(X, y, split_ratio=0.8)
    lgbm_results = train_lightgbm_model(X, y, split_ratio=0.8)
    
    print("\n--- Model Benchmark Comparison ---")
    print(f"Baseline (Higher Elo Wins): Acc = {rf_results['baseline']['accuracy']*100:.2f}%, ROC AUC = {rf_results['baseline']['roc_auc']:.4f}")
    print(f"Random Forest Classifier:    Acc = {rf_results['metrics']['accuracy']*100:.2f}%, ROC AUC = {rf_results['metrics']['roc_auc']:.4f}")
    print(f"LightGBM Classifier:         Acc = {lgbm_results['metrics']['accuracy']*100:.2f}%, ROC AUC = {lgbm_results['metrics']['roc_auc']:.4f}")
    
    # 5. Train Victory Method Multi-class Classifier
    print("\n[5/6] Training Victory Method Multi-Class Classifier (Ippon vs Wazari vs Penalties)...")
    y_method = parse_victory_method_target(df_clean)
    method_results = train_victory_method_model(X, y_method, split_ratio=0.8)
    print(f"Victory Method Model Accuracy: {method_results['metrics']['accuracy']*100:.2f}%")
    
    # 6. Generate Visualizations
    print(f"\n[6/6] Generating analytical plots in: {PLOTS_DIR}")
    generate_and_save_plots(df_elo, final_ratings, rf_results, PLOTS_DIR)
    
    print("\n==================================================")
    print("   PIPELINE EXECUTED SUCCESSFULLY!")
    print("   Launch Web Dashboard: uv run streamlit run app.py")
    print("==================================================")

if __name__ == "__main__":
    main()
