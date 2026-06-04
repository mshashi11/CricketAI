#!/usr/bin/python3
"XGBoost Win Probability and Score Projection (with Volatility-based Bounds) for T10 format"

import os
import urllib.request
import pandas as pd
import numpy as np
import xgboost as xgb

def ensure_data_exists() -> str:
    """Checks for local CSV data; downloads from remote host if missing.
    Returns the path to the directory containing the data files."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    remote_files = {
        "t10_inn1_data.csv": "https://github.com/mshashi11/CricketAI/releases/download/v1.0.0/t10_inn1_data.csv",
        "t10_inn2_data.csv": "https://github.com/mshashi11/CricketAI/releases/download/v1.0.0/t10_inn2_data.csv"
    }

    for filename, url in remote_files.items():
        file_path = os.path.join(script_dir, filename)
        if not os.path.exists(file_path):
            print(f"'{filename}' not found at {file_path}. Downloading from GitHub Releases...")
            try:
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                urllib.request.urlretrieve(url, file_path)
                print(f"Successfully downloaded '{filename}'.")
            except Exception as e:
                raise IOError(f"Failed to download {filename} from {url}. Error: {e}")
    return script_dir

def train_xgb_models():
    # Ensure dataset is downloaded and get directory path
    data_dir = ensure_data_exists()

    print("Loading and training 1st Inning models...")
    df1 = pd.read_csv(os.path.join(data_dir, "t10_inn1_data.csv"))
    
    X1 = df1[['cum_balls', 'cum_wickets', 'cum_runs']]
    y_wp1 = df1['won']
    y_score1 = df1['total_score']
    
    # 1. WP1 Model: Win Probability
    wp_model_inn1 = xgb.XGBClassifier(
        n_estimators=1000, max_depth=5, learning_rate=0.05,
        monotone_constraints=(-1, -1, 1), objective='binary:logistic'
    )
    wp_model_inn1.fit(X1, y_wp1)
    
    # 2. Mean Score Model
    score_model_mean = xgb.XGBRegressor(
        n_estimators=1000, max_depth=5, learning_rate=0.05,
        monotone_constraints=(-1, -1, 1), objective='reg:squarederror'
    )
    score_model_mean.fit(X1, y_score1)

    # 3. Volatility Model (Predicting the Absolute Error)
    print("Training Volatility model...")
    # Get residuals from the mean model
    mean_preds_train = score_model_mean.predict(X1)
    abs_error = np.abs(y_score1 - mean_preds_train)
    
    # Train model to predict the expected volatility (spread) at any given state
    # We apply monotonic constraints here too: uncertainty generally decreases as balls/wickets run out
    score_model_vol = xgb.XGBRegressor(
        n_estimators=1000, max_depth=5, learning_rate=0.05,
        monotone_constraints=(-1, -1, 0), # Volatility decreases with balls and wickets, runs is neutral
        objective='reg:squarederror'
    )
    score_model_vol.fit(X1, abs_error)
    
    print("Loading and training 2nd Inning models...")
    df2 = pd.read_csv(os.path.join(data_dir, "t10_inn2_data.csv"))
    X2 = df2[['rem_balls', 'wickets_hand', 'runs_chase']]
    y_wp2 = df2['won']
    
    wp_model_inn2 = xgb.XGBClassifier(
        n_estimators=1000, max_depth=5, learning_rate=0.05,
        monotone_constraints=(1, 1, -1), objective='binary:logistic'
    )
    wp_model_inn2.fit(X2, y_wp2)
    
    return wp_model_inn1, score_model_mean, score_model_vol, wp_model_inn2

def generate_tables(wp_model_inn1, score_model_mean, score_model_vol, wp_model_inn2):
    max_balls, max_wickets, max_runs = 60, 10, 200
    k = 1.15 # Multiplier for 75% range (approx 1.15 * mean absolute error for normal distribution)
    
    print("Generating Inn1 Table...")
    grid = []
    for balls in range(max_balls + 1):
        for wickets in range(max_wickets + 1):
            for runs in range(max_runs + 1):
                grid.append([balls, wickets, runs])
    grid = np.array(grid)
    
    X_inn1 = pd.DataFrame(grid, columns=['cum_balls', 'cum_wickets', 'cum_runs'])
    wp_probs1 = wp_model_inn1.predict_proba(X_inn1)[:, 1]
    mean_preds = score_model_mean.predict(X_inn1)
    vol_preds = score_model_vol.predict(X_inn1)
    
    # Calculate bounds based on predicted volatility
    score_low = mean_preds - (k * vol_preds)
    score_high = mean_preds + (k * vol_preds)
    
    # Apply Boundary Constraints
    mask_balls_60 = (grid[:, 0] == 60)
    runs_at_60 = grid[mask_balls_60, 2]
    
    # WP Override at Break
    X_break = pd.DataFrame({'rem_balls': [60] * len(runs_at_60), 'wickets_hand': [10] * len(runs_at_60), 'runs_chase': runs_at_60 + 1})
    wp_probs_inn2_break = wp_model_inn2.predict_proba(X_break)[:, 1]
    wp_probs1[mask_balls_60] = 1.0 - wp_probs_inn2_break
    
    # Score/Bounds Override at Break
    mean_preds[mask_balls_60] = runs_at_60
    score_low[mask_balls_60] = runs_at_60
    score_high[mask_balls_60] = runs_at_60
    
    # Score/Bounds Override at All Out
    mask_wickets_10 = (grid[:, 1] == 10)
    runs_at_w10 = grid[mask_wickets_10, 2]
    mean_preds[mask_wickets_10] = runs_at_w10
    score_low[mask_wickets_10] = runs_at_w10
    score_high[mask_wickets_10] = runs_at_w10
    
    # Ensure logical sanity
    score_low = np.maximum(score_low, grid[:, 2]) # Cannot project less than current runs
    score_low = np.minimum(score_low, mean_preds)
    score_high = np.maximum(score_high, mean_preds)

    print("Writing probs_i1_t10.txt...")
    with open("probs_i1_t10.txt", "w") as f:
        for i in range(len(grid)):
            f.write(f"{int(grid[i,0])}\t{int(grid[i,1])}\t{int(grid[i,2])}\t{wp_probs1[i]:.6f}\t{mean_preds[i]:.1f}\t{score_low[i]:.1f}\t{score_high[i]:.1f}\n")

    print("Generating Inn2 Table...")
    grid2 = np.array(grid)
    X_inn2 = pd.DataFrame(grid2, columns=['rem_balls', 'wickets_hand', 'runs_chase'])
    wp_probs2 = wp_model_inn2.predict_proba(X_inn2)[:, 1]
    
    with open("probs_i2_t10.txt", "w") as f:
        for i in range(len(grid2)):
            f.write(f"{int(grid2[i,0])}\t{int(grid2[i,1])}\t{int(grid2[i,2])}\t{wp_probs2[i]:.6f}\n")

def main():
    models = train_xgb_models()
    generate_tables(*models)
    print("Volatility-based XGBoost tables generated successfully.")

if __name__ == "__main__":
    main()
