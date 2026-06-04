# CricketAI: T10 Win Probability Models

This repository contains statistical and Machine Learning models to compute **Win Probability (WP)** and **Projected Inning Scores** for the **T10 format** of Cricket.

The project contains two main parts:
1. **Data Prep** (`DataPrep/`): Script to extract ball-by-ball data from a match database and generate structured training datasets.
2. **Win Probability Modeling** (`WinProbability/`): Deep Learning (PyTorch) and Gradient Boosting (XGBoost) models to train and generate win probability state tables.

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
The training scripts are located in the `WinProbability/` directory:
- **PyTorch Neural Network**: [wp_model_nn.py](file:///home/shanuja/CricketAI/WinProbability/wp_model_nn.py)
- **XGBoost Regressor/Classifier**: [wp_model_xgb.py](file:///home/shanuja/CricketAI/WinProbability/wp_model_xgb.py)

To run either script:
```bash
python WinProbability/wp_model_nn.py
# OR
python WinProbability/wp_model_xgb.py
```

> [!NOTE]
> **Zero Manual Data Download Required:** You do not need to download the dataset files manually. The training scripts contain built-in helper functions (`ensure_data_exists()`) that check for the CSV files locally and automatically fetch them from the GitHub Release assets if they are missing.

---

## Dataset Hosting via GitHub Releases

To keep the repository lightweight and avoid tracking huge files under Git version control, the datasets `t10_inn1_data.csv` and `t10_inn2_data.csv` are ignored by Git (via `.gitignore`) and hosted as GitHub Release Assets.

### How to release a new version of the datasets:
If you regenerate or update the datasets using the `DataPrep` tools, follow these steps to release them:

1. **Tag and Create a GitHub Release**:
   - Go to your repository page: `https://github.com/mshashi11/CricketAI`
   - Click on **Releases** -> **Draft a new release**.
   - Create a tag named `v1.0.0` (this matches the default download tag in the scripts).
2. **Attach the Datasets**:
   - Drag and drop your newly generated `t10_inn1_data.csv` and `t10_inn2_data.csv` files into the **Attach binaries** field.
   - Publish the release.
3. **Automatic Fetching**:
   - Once published, anyone running the training scripts will automatically download the correct version of the datasets directly from your release assets.

---

## Repository Structure

- `WinProbability/`:
  - [wp_model_nn.py](file:///home/shanuja/CricketAI/WinProbability/wp_model_nn.py): PyTorch NN implementation. Trains models for the 1st inning, 2nd inning, and 1st inning score projections, enforcing innings-break boundary consistency.
  - [wp_model_xgb.py](file:///home/shanuja/CricketAI/WinProbability/wp_model_xgb.py): XGBoost implementation with volatility-based bounds.
- `DataPrep/`:
  - [generate_t10_data.py](file:///home/shanuja/CricketAI/DataPrep/generate_t10_data.py): Script that connects to the Cricmetric database and processes ball-by-ball inputs into modeling rows.
  - [db_connections.py](file:///home/shanuja/CricketAI/DataPrep/db_connections.py): Handles local DB connections (database credentials/connections are ignored from Git).
