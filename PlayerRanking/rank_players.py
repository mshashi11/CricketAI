"""
Module for ranking cricket players using the Eigen Factor (EFscore) algorithm.
This is a Markov chain based ranking system that evaluates players based on
their pairwise interactions.
"""

import os
import urllib.request
import numpy as np
import pandas as pd

def get_rankings(matchups):
    """
    Computes player rankings based on pairwise interaction results.

    Args:
        matchups: A pandas DataFrame containing:
            'Batsman', 'Bowler', 'Dots', 'Singles', 'Doubles',
            'Triples', 'Fours', 'Sixes', 'Outs'

    Returns:
        A pandas DataFrame with Player, EFscore, Wins, Losses, Ties, and EVrank
    """
    # Combine batsmen and bowlers to get all unique players
    players = sorted(list(set(matchups['Batsman']) | set(matchups['Bowler'])))
    num_players = len(players)
    player_to_idx = {player: i for i, player in enumerate(players)}

    # Calculate runs scored in each interaction
    matchups = matchups.copy()
    matchups['batsman_runs'] = (
        matchups['Singles'] +
        2 * matchups['Doubles'] +
        3 * matchups['Triples'] +
        4 * matchups['Fours'] +
        6 * matchups['Sixes']
    )

    # Compute overall statistics
    total_runs = matchups['batsman_runs'].sum()
    total_outs = matchups['Outs'].sum()

    # Batting average (runs per out)
    batting_average = total_runs / total_outs if total_outs > 0 else 0

    total_balls = (
        matchups['Dots'].sum() +
        matchups['Singles'].sum() +
        matchups['Doubles'].sum() +
        matchups['Triples'].sum() +
        matchups['Fours'].sum() +
        matchups['Sixes'].sum() +
        matchups['Outs'].sum()
    )

    # Runs per ball
    runs_per_ball = total_runs / total_balls if total_balls > 0 else 0

    # Calculate wins/losses for each matchup row
    # For dot balls and wickets, the bowler is considered victorious
    matchups['bowler_score'] = runs_per_ball * matchups['Dots']

    # Compute the wicket score
    matchups['balls'] = (
        matchups['Dots'] + matchups['Singles'] + matchups['Doubles'] +
        matchups['Triples'] + matchups['Fours'] + matchups['Sixes'] + matchups['Outs']
    )

    # wicket_score = runs_per_ball * balls - outs * batting_average
    matchups['wicket_score'] = (matchups['balls'] * runs_per_ball -
                               matchups['Outs'] * batting_average)

    # If wicket_score > 0 then this is added to batsman score, otherwise added to bowler score
    matchups['batsman_score'] = np.where(matchups['wicket_score'] > 0,
                                         matchups['wicket_score'], 0)
    matchups['bowler_score'] += np.where(matchups['wicket_score'] < 0,
                                         -matchups['wicket_score'], 0)

    # For singles, batsman is victorious if runs_per_ball < 1
    if runs_per_ball < 1:
        matchups['batsman_score'] += (1 - runs_per_ball) * matchups['Singles']
    else:
        matchups['bowler_score'] += (runs_per_ball - 1) * matchups['Singles']

    # For higher boundaries, batsman is almost always victorious (unless runs_per_ball > 2, 3, etc)
    matchups['batsman_score'] += (
        (2 - runs_per_ball) * matchups['Doubles'] +
        (3 - runs_per_ball) * matchups['Triples'] +
        (4 - runs_per_ball) * matchups['Fours'] +
        (6 - runs_per_ball) * matchups['Sixes']
    )

    # Aggregate scores by Batsman and Bowler pairs
    aggregated = matchups.groupby(['Batsman', 'Bowler']).agg({
        'batsman_score': 'sum',
        'bowler_score': 'sum'
    }).reset_index()

    # Create the pairwise win-loss matrix
    # win_loss_matrix[i, j] corresponds to the number of times player i
    # has been defeated by player j
    win_loss_matrix = np.zeros((num_players, num_players))

    for _, row in aggregated.iterrows():
        b_idx = player_to_idx[row['Batsman']]
        p_idx = player_to_idx[row['Bowler']]

        # Bowler j defeats Batsman i
        win_loss_matrix[b_idx, p_idx] += row['bowler_score']
        # Batsman i defeats Bowler j
        win_loss_matrix[p_idx, b_idx] += row['batsman_score']

    # Add 0.001 to ensure the Markov chain is recurrent
    transition_matrix = win_loss_matrix + 0.001

    # Set diagonal entries to total wins of the player
    for i in range(num_players):
        transition_matrix[i, i] = np.sum(transition_matrix[:, i])

    # Normalize each row to get a stochastic matrix
    row_sums = transition_matrix.sum(axis=1)
    transition_matrix = transition_matrix / row_sums[:, np.newaxis]

    # Compute the principal eigenvector of the transpose of the transition matrix
    eigenvalues, eigenvectors = np.linalg.eig(transition_matrix.T)

    # The eigenvalue closest to 1.0 is the one we want
    idx = np.argmin(np.abs(eigenvalues - 1.0))
    ef_score = np.abs(eigenvectors[:, idx])

    # Create the result DataFrame
    results = pd.DataFrame({
        'Player': players,
        'EFscore': ef_score
    })

    # Add wins and losses (aggregated from win_loss_matrix)
    results['Wins'] = win_loss_matrix.sum(axis=0)
    results['Losses'] = win_loss_matrix.sum(axis=1)
    results['Ties'] = 0.0

    # Sort by EFscore descending
    results = results.sort_values(by='EFscore', ascending=False)
    results['EVrank'] = range(1, num_players + 1)

    return results

def ensure_data_exists() -> str:
    """Checks for local CSV data; downloads from remote host if missing.
    Returns the path to the directory containing the data files."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    remote_files = {
        "matchups_ipl2026.csv": (
            "https://github.com/mshashi11/CricketAI/"
            "releases/download/v1.0.0/matchups_ipl2026.csv"
        )
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
                raise IOError(
                    f"Failed to download {filename} from {url}. Error: {e}"
                ) from e
    return script_dir

def main():
    """Main function to run player ranking computation for IPL 2026."""
    data_dir = ensure_data_exists()
    matchup_path = os.path.join(data_dir, "matchups_ipl2026.csv")
    print(f"Loading matchup data from {matchup_path}...")
    matchups = pd.read_csv(matchup_path)

    print("Computing rankings...")
    ranks = get_rankings(matchups)

    # Calculate RAA (Runs Above Average)
    ranks['RAA'] = ranks['Wins'] - ranks['Losses']

    # Calculate overall batting average
    total_runs = (
        matchups['Singles'] +
        2 * matchups['Doubles'] +
        3 * matchups['Triples'] +
        4 * matchups['Fours'] +
        6 * matchups['Sixes']
    ).sum()
    total_outs = matchups['Outs'].sum()

    overall_average = total_runs / total_outs if total_outs > 0 else 0

    # Calculate Wins contributed
    if overall_average > 0:
        ranks['Wins_contributed'] = ranks['RAA'] / (10 * overall_average)
    else:
        ranks['Wins_contributed'] = 0.0

    ranks['series_id'] = 'ipl2026'

    # Select and rename columns to match database schema
    output_df = ranks[['Player', 'EFscore', 'RAA', 'Wins_contributed', 'series_id']].copy()
    output_df.columns = ['player_id', 'EFscore', 'RAA', 'Wins', 'series_id']

    # Sort in decreasing order of EF score
    output_df = output_df.sort_values(by='EFscore', ascending=False)

    # Print the ranks to a text file
    output_file = os.path.join(data_dir, "ranks_ipl2026.txt")
    output_df.to_csv(output_file, sep='\t', index=False)
    print(f"Rankings saved to {output_file}")

if __name__ == "__main__":
    main()
