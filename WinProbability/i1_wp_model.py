#!/usr/bin/python3
"Win Probability Models for T10 format (1st and 2nd Inning)"

import torch
import pandas as pd
from torch import nn

class CricketWPModel(nn.Module):
    """Class that defines the deep learning model for Win Probability"""

    def __init__(self, dim: int = 20):
        """
        Constructor for the class
        Parameters:
          dim: Dimension of the hidden layer of the network
        """
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(in_features=3, out_features=dim),
            nn.ReLU(),
            nn.Linear(in_features=dim, out_features=1)
        )

    def forward(self, x: torch.Tensor):
        return self.network(x)

def train_model(
        model: CricketWPModel,
        data_x: torch.Tensor,
        data_y: torch.Tensor,
        criterion: nn.Module,
        num_iter: int = 2000,
        lr: float = 0.01
) -> None:
    """Train the given deep learning model"""
    optim = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_iter):
        # Clear the old gradients
        optim.zero_grad()

        # Forward pass
        y_pred = model(data_x)

        # Compute the loss
        loss = criterion(y_pred, data_y)

        # Backward propagation
        loss.backward()

        # Gradient update
        optim.step()

        if epoch % 200 == 0:
            print(f"Epoch: {epoch} | Loss: {loss:.4f}")

def generate_inn1_wp_table(
        wp_model: CricketWPModel,
        score_model: CricketWPModel,
        wp_model_inn2: CricketWPModel,
        device: torch.device,
        max_balls: int = 60,
        max_runs: int = 200,
) -> torch.Tensor:
    """Generate the full win probability table for the 1st inning"""
    rows = []
    for balls in range(max_balls + 1):
        for wickets in range(11):
            for runs in range(max_runs + 1):
                rows.append([balls, wickets, runs])

    model_input = torch.tensor(rows).float().to(device)

    wp_model.eval()
    score_model.eval()    
    wp_model_inn2.eval()

    with torch.no_grad():
        # Get the full win probability evaluation in one go
        wp_model_output = torch.sigmoid(wp_model(model_input))
        score_model_output = score_model(model_input)
        
        # --- Override for End of Inning (balls == 60) ---
        # Condition: WP_Inn1(60, w, r) = 1 - WP_Inn2(60, 10, r+1)
        # This ensures the core requirement is met in the output table.
        
        # Identify rows where balls == 60
        mask_end_overs = (model_input[:, 0] == max_balls)
        
        if mask_end_overs.any():
            runs_at_end = model_input[mask_end_overs, 2] # Extract runs column
            
            # Prepare inputs for Inn2 model: (60, 10, runs+1)
            # rem_balls = 60, wickets_hand = 10, runs_chase = runs + 1
            
            inn2_input = torch.stack([
                torch.full_like(runs_at_end, max_balls), # rem_balls = 60
                torch.full_like(runs_at_end, 10),        # wickets_hand = 10
                runs_at_end + 1                          # runs_chase
            ], dim=1).float().to(device)
            
            inn2_output = torch.sigmoid(wp_model_inn2(inn2_input))
            
            # Apply override: WP1 = 1 - WP2
            wp_model_output[mask_end_overs] = 1.0 - inn2_output

            # Override projected score at end of inning
            score_model_output[mask_end_overs] = runs_at_end.unsqueeze(1)

        # --- Override for All Out (wickets == 10) ---
        mask_all_out = (model_input[:, 1] == 10)
        if mask_all_out.any():
             runs_at_wickets = model_input[mask_all_out, 2]
             score_model_output[mask_all_out] = runs_at_wickets.unsqueeze(1)


    # Return tensor with the inputs and outputs stacked columnwise
    return torch.column_stack((torch.tensor(rows).to(device), wp_model_output, score_model_output))

def generate_inn2_wp_table(
        wp_model: CricketWPModel,
        device: torch.device,
        max_balls: int = 60,
        max_runs_chase: int = 200,
) -> torch.Tensor:
    """Generate the full win probability table for the 2nd inning"""
    rows = []
    for balls in range(max_balls + 1):
        for wickets in range(11):
            for runs in range(max_runs_chase + 1):
                rows.append([balls, wickets, runs])
    
    model_input = torch.tensor(rows).float().to(device)
    wp_model.eval()
    
    with torch.no_grad():
        wp_model_output = torch.sigmoid(wp_model(model_input))
        
    return torch.column_stack((torch.tensor(rows).to(device), wp_model_output))

def main():
    """Main function, the execution starts here"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- 1st Inning Data ---
    print("Loading 1st Inning Data...")
    training_data_inn1 = pd.read_csv("t10_inn1_data.csv")
    data_x_inn1 = torch.from_numpy(training_data_inn1[['cum_balls', 'cum_wickets', 'cum_runs']].values).float().to(device)
    data_y_inn1 = torch.from_numpy(training_data_inn1[['won']].values).float().to(device)
    data_score_inn1 = torch.from_numpy(training_data_inn1[['total_score']].values).float().to(device)

    # --- Train 1st Inning WP Model ---
    print("Training 1st Inning WP Model...")
    wp_model_inn1 = CricketWPModel().to(device)
    train_model(wp_model_inn1, data_x_inn1, data_y_inn1, nn.BCEWithLogitsLoss())

    # --- Train 1st Inning Score Model ---
    print("Training 1st Inning Score Model...")
    score_model_inn1 = CricketWPModel().to(device)
    train_model(score_model_inn1, data_x_inn1, data_score_inn1, nn.MSELoss())

    # --- 2nd Inning Data ---
    print("Loading 2nd Inning Data...")
    training_data_inn2 = pd.read_csv("t10_inn2_data.csv")
    data_x_inn2 = torch.from_numpy(training_data_inn2[['rem_balls', 'wickets_hand', 'runs_chase']].values).float().to(device)
    data_y_inn2 = torch.from_numpy(training_data_inn2[['won']].values).float().to(device)

    # --- Train 2nd Inning WP Model ---
    print("Training 2nd Inning WP Model...")
    wp_model_inn2 = CricketWPModel().to(device)
    train_model(wp_model_inn2, data_x_inn2, data_y_inn2, nn.BCEWithLogitsLoss())

    # --- Verification ---
    print("\nVerifying consistency at Innings Break (Raw Models)...")
    test_runs = [80, 100, 120, 140]
    test_wickets = [2, 5]
    
    for w in test_wickets:
        for r in test_runs:
            # Inn1 prediction
            inp1 = torch.tensor([[60, w, r]]).float().to(device)
            wp1 = torch.sigmoid(wp_model_inn1(inp1)).item()
            
            # Inn2 prediction
            inp2 = torch.tensor([[60, 10, r+1]]).float().to(device)
            wp2 = torch.sigmoid(wp_model_inn2(inp2)).item()
            
            print(f"Runs: {r}, Wickets: {w} -> WP1: {wp1:.4f}, WP2: {wp2:.4f}, Sum: {wp1+wp2:.4f}")

    # --- Generate Tables ---
    print("\nGenerating 1st Inning WP Table (with consistency override)...")
    wp_table_inn1 = generate_inn1_wp_table(wp_model_inn1, score_model_inn1, wp_model_inn2, device)

    # Write Inn1 table
    with open("probs_i1_t10.txt", "w") as f:
        for row in wp_table_inn1:
            row = list(row)
            f.write(f"{int(row[0])}\t{int(row[1])}\t{int(row[2])}\t{row[3]:.6f}\t{row[4]:.0f}\n")
    
    print("Generating 2nd Inning WP Table...")
    wp_table_inn2 = generate_inn2_wp_table(wp_model_inn2, device)

    # Write Inn2 table
    with open("probs_i2_t10.txt", "w") as f:
        for row in wp_table_inn2:
            row = list(row)
            # balls, wickets, runs_chase, wp
            f.write(f"{int(row[0])}\t{int(row[1])}\t{int(row[2])}\t{row[3]:.6f}\n")

    print("Done.")

if __name__ == "__main__":
    main()
