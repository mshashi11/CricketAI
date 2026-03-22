#!/usr/bin/python3
"XGBoost Win Probability and Score Projection (with 95% Bounds) for T10 format"

import pandas as pd
import numpy as np
import xgboost as xgb

def train_xgb_models():
    print("Loading and training 1st Inning models...")
    df1 = pd.read_csv("t10_inn1_data.csv")
    
    X1 = df1[['cum_balls', 'cum_wickets', 'cum_runs']]
    y_wp1 = df1['won']
    y_score1 = df1['total_score']
    
    # WP1 Model: Win Probability
    wp_model_inn1 = xgb.XGBClassifier(
        n_estimators=1000, max_depth=5, learning_rate=0.05,
        monotone_constraints=(-1, -1, 1), objective='binary:logistic'
    )
    wp_model_inn1.fit(X1, y_wp1)
    
    # Mean Score Model
    score_model_inn1 = xgb.XGBRegressor(
        n_estimators=1000, max_depth=5, learning_rate=0.05,
        monotone_constraints=(-1, -1, 1), objective='reg:squarederror'
    )
    score_model_inn1.fit(X1, y_score1)

    # Low Bound (12.5th percentile)
    score_model_low = xgb.XGBRegressor(
        n_estimators=1000, max_depth=5, learning_rate=0.05,
        monotone_constraints=(-1, -1, 1), 
        objective='reg:quantileerror', quantile_alpha=0.125
    )
    score_model_low.fit(X1, y_score1)

    # High Bound (87.5th percentile)
    score_model_high = xgb.XGBRegressor(
        n_estimators=1000, max_depth=5, learning_rate=0.05,
        monotone_constraints=(-1, -1, 1),
        objective='reg:quantileerror', quantile_alpha=0.875
    )
    score_model_high.fit(X1, y_score1)
    
    print("Loading and training 2nd Inning models...")
    df2 = pd.read_csv("t10_inn2_data.csv")
    X2 = df2[['rem_balls', 'wickets_hand', 'runs_chase']]
    y_wp2 = df2['won']
    
    wp_model_inn2 = xgb.XGBClassifier(
        n_estimators=1000, max_depth=5, learning_rate=0.05,
        monotone_constraints=(1, 1, -1), objective='binary:logistic'
    )
    wp_model_inn2.fit(X2, y_wp2)
    
    return wp_model_inn1, score_model_inn1, score_model_low, score_model_high, wp_model_inn2

def generate_tables(wp_model_inn1, score_model_inn1, score_model_low, score_model_high, wp_model_inn2):
    max_balls, max_wickets, max_runs = 60, 10, 200
    
    print("Generating Inn1 Table...")
    grid = []
    for balls in range(max_balls + 1):
        for wickets in range(max_wickets + 1):
            for runs in range(max_runs + 1):
                grid.append([balls, wickets, runs])
    grid = np.array(grid)
    
    X_inn1 = pd.DataFrame(grid, columns=['cum_balls', 'cum_wickets', 'cum_runs'])
    wp_probs1 = wp_model_inn1.predict_proba(X_inn1)[:, 1]
    score_preds1 = score_model_inn1.predict(X_inn1)
    score_low = score_model_low.predict(X_inn1)
    score_high = score_model_high.predict(X_inn1)
    
    # Apply Boundary Constraints
    mask_balls_60 = (grid[:, 0] == 60)
    runs_at_60 = grid[mask_balls_60, 2]
    
    X_break = pd.DataFrame({'rem_balls': [60] * len(runs_at_60), 'wickets_hand': [10] * len(runs_at_60), 'runs_chase': runs_at_60 + 1})
    wp_probs_inn2_break = wp_model_inn2.predict_proba(X_break)[:, 1]
    
    wp_probs1[mask_balls_60] = 1.0 - wp_probs_inn2_break
    score_preds1[mask_balls_60] = runs_at_60
    score_low[mask_balls_60] = runs_at_60
    score_high[mask_balls_60] = runs_at_60
    
    mask_wickets_10 = (grid[:, 1] == 10)
    score_preds1[mask_wickets_10] = grid[mask_wickets_10, 2]
    score_low[mask_wickets_10] = grid[mask_wickets_10, 2]
    score_high[mask_wickets_10] = grid[mask_wickets_10, 2]
    
    # Ensure logical ordering (Low <= Mean <= High)
    score_low = np.minimum(score_low, score_preds1)
    score_high = np.maximum(score_high, score_preds1)

    print("Writing probs_i1_t10.txt...")
    with open("probs_i1_t10.txt", "w") as f:
        for i in range(len(grid)):
            f.write(f"{int(grid[i,0])}\t{int(grid[i,1])}\t{int(grid[i,2])}\t{wp_probs1[i]:.6f}\t{score_preds1[i]:.1f}\t{score_low[i]:.1f}\t{score_high[i]:.1f}\n")

    print("Generating Inn2 Table...")
    grid2 = np.array(grid) # Same grid for 2nd inning
    X_inn2 = pd.DataFrame(grid2, columns=['rem_balls', 'wickets_hand', 'runs_chase'])
    wp_probs2 = wp_model_inn2.predict_proba(X_inn2)[:, 1]
    
    with open("probs_i2_t10.txt", "w") as f:
        for i in range(len(grid2)):
            f.write(f"{int(grid2[i,0])}\t{int(grid2[i,1])}\t{int(grid2[i,2])}\t{wp_probs2[i]:.6f}\n")

def main():
    models = train_xgb_models()
    generate_tables(*models)
    print("XGBoost tables with 95% confidence bounds generated successfully.")

if __name__ == "__main__":
    main()
