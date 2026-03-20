#!/usr/bin/python3
"1st Inning Win Probability Model for T10 format"

import torch
import pandas as pd
from torch import nn


class FirstInnWPmodel(nn.Module):
    "Class that defines the deep learning model for 1st Inning Win Probability"

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


def train_wp_model(
        model: FirstInnWPmodel,
        data_x: torch.Tensor,
        data_y: torch.Tensor,
        num_iter: int = 100
) -> None:
    "Train the given deep learning model"
    optim = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.BCEWithLogitsLoss()

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

        if epoch % 10 == 0:
            print(f"Epoch: {epoch} | Loss: {loss:.2f}")


def train_i1_score_model(
        model: FirstInnWPmodel,
        data_x: torch.Tensor,
        data_y: torch.Tensor,
        num_iter: int = 100
) -> None:
    "Train the given deep learning model"
    optim = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()

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

        if epoch % 10 == 0:
            print(f"Epoch: {epoch} | Loss: {loss:.2f}")


def generate_wp_table(
        wp_model: FirstInnWPmodel,
        i1_score_model: FirstInnWPmodel,
        device: torch.device,
        max_balls: int = 60,
        max_runs: int = 200,
) -> torch.Tensor:
    "Generate the full win probability table with the given model"
    rows = []
    for balls in range(max_balls + 1):
        for wickets in range(11):
            for runs in range(max_runs + 1):
                rows.append([balls, wickets, runs])

    model_input = torch.tensor(rows).float().to(device)

    wp_model.eval()
    i1_score_model.eval()    

    with torch.no_grad():
        # Get the full win probability evaluation in one go
        wp_model_output = torch.sigmoid(wp_model(model_input))
        i1_score_model_output = i1_score_model(model_input)

    # Return tensor with the inputs and outputs stacked columnwise
    return torch.column_stack((torch.tensor(rows).to(device), wp_model_output, i1_score_model_output))


def main():
    "Main function, the execution starts here"
    device = torch.device("cuda")

    # Read in the training data
    training_data = pd.read_csv("t10_inn1_data.csv")
    data_x = torch.from_numpy(training_data[['cum_balls', 'cum_wickets', 'cum_runs']].values).float().to(device)
    data_y = torch.from_numpy(training_data[['won']].values).float().to(device)
    i1_score = torch.from_numpy(training_data[['total_score']].values).float().to(device)

    wp_model = FirstInnWPmodel().to(device)
    train_wp_model(wp_model, data_x, data_y)

    i1_score_model = FirstInnWPmodel().to(device)
    train_i1_score_model(i1_score_model, data_x, i1_score)

    wp_table = generate_wp_table(wp_model, i1_score_model, device)

    # Print the WP table to a text file
    with open("probs_i1_t10.txt", "w") as f:
        for row in wp_table:
            row = list(row)
            f.write(f"{int(row[0])}\t{int(row[1])}\t{int(row[2])}\t{row[3]:.6f}\t{row[4]:.0f}\n")


if __name__ == "__main__":
    main()
