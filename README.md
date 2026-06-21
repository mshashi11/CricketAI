# CricketAI: Building AI models with Cricket data

This repository contains statistical and Machine Learning models using ball by ball data from Cricket matches.

The project contains the following parts:

1. **Win Probability Modeling** (`WinProbability/`): Deep Learning (PyTorch) and Gradient Boosting (XGBoost) models to train and generate win probability state tables.
2. **Player Ranking Model** (`PlayerRanking/`): A Markov Chain (PageRank-style) algorithm to rank players based on ball-by-ball pairwise matchups (batsman vs. bowler).

---

## Getting Started

### 1. Installation
Clone the repository and install the dependencies:
```bash
git clone https://github.com/mshashi11/CricketAI.git
cd CricketAI
pip install -r requirements.txt
```
*(Make sure you have PyTorch, XGBoost, pandas, numpy, and scikit-learn installed)*

### 2. Running the Models
The training and ranking scripts are located in their respective folders:
- **Win Probability Models**:
  - **PyTorch Neural Network**: [wp_model_nn.py](file:///Users/shanuja/CricketAI/WinProbability/wp_model_nn.py)
  - **XGBoost Regressor/Classifier**: [wp_model_xgb.py](file:///Users/shanuja/CricketAI/WinProbability/wp_model_xgb.py)
- **Player Ranking Model**:
  - **EigenFactor Player Ranking**: [rank_players.py](file:///Users/shanuja/CricketAI/PlayerRanking/rank_players.py)

To run the scripts:
```bash
# Win Probability Neural Network
python3 WinProbability/wp_model_nn.py

# Win Probability XGBoost
python3 WinProbability/wp_model_xgb.py

# Player Ranking Model
python3 PlayerRanking/rank_players.py
```

> [!NOTE]
> **Zero Manual Data Download Required:** You do not need to download the dataset/matchup files manually. The scripts contain built-in helper functions (`ensure_data_exists()`) that check for the CSV files locally and automatically fetch them from the GitHub Release assets if they are missing.

---

## Dataset Hosting via GitHub Releases

To keep the repository lightweight and avoid tracking huge files under Git version control, the datasets `t10_inn1_data.csv`, `t10_inn2_data.csv`, and `matchups_ipl2026.csv` are ignored by Git (via `.gitignore`) and hosted as GitHub Release Assets.

### How to release a new version of the datasets:
If you regenerate or update the datasets, follow these steps to release them:

1. **Tag and Create a GitHub Release**:
   - Go to your repository page: `https://github.com/mshashi11/CricketAI`
   - Click on **Releases** -> **Draft a new release**.
   - Create a tag named `v1.0.0` (this matches the default download tag in the scripts).
2. **Attach the Datasets**:
   - Drag and drop your newly generated `t10_inn1_data.csv`, `t10_inn2_data.csv`, and `matchups_ipl2026.csv` files into the **Attach binaries** field.
   - Publish the release.
3. **Automatic Fetching**:
   - Once published, anyone running the scripts will automatically download the correct version of the datasets directly from your release assets.

---

## Repository Structure

- `WinProbability/`:
  - [wp_model_nn.py](file:///Users/shanuja/CricketAI/WinProbability/wp_model_nn.py): PyTorch NN implementation. Trains models for the 1st inning, 2nd inning, and 1st inning score projections, enforcing innings-break boundary consistency.
  - [wp_model_xgb.py](file:///Users/shanuja/CricketAI/WinProbability/wp_model_xgb.py): XGBoost implementation with volatility-based bounds.
- `PlayerRanking/`:
  - [rank_players.py](file:///Users/shanuja/CricketAI/PlayerRanking/rank_players.py): PageRank-style Markov chain ranking algorithm implementation. Calculates player EigenFactor scores, Runs Above Average (RAA), and Wins Contributed based on batsman-bowler matchups.
