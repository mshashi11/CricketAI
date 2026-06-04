The script wp_model_nn.py contains the code for win probability model for the T10 format in Cricket. Currently it contains the code for computing the win probability for the 1st inning only. Here are the additional things that I want to accomplish:

1. Projected 1st inning score
2. Win probability for the second inning, for the team batting second

The goal is that the win probability at the end of the 1st inning for the team batting first should match the win probability of that team at the start of the second inning. This is a core requirement for the win probability model.

Data sets available:

1. t10_inn1_data.csv: CSV file, fields are:
   - Cumulative balls
   - Cumulative innings
   - Cumulative runs
   - Total 1st inning score for this match
   - Whether the team batting 1st won the match or not

2. t10_inn2_data.csv: CSV file, fields are:
   - Remaining balls in the second inning
   - Remaining wickets in hand in the second inning
   - Runs to be chased
   - Whether the team batting 2nd won the match or not