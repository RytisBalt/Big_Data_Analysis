import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import poisson

# ---------------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------------

MAX_GOALS = 8

GROUP_ASSIGNMENTS = {
    'Group A': ['Mexico', 'South Africa', 'South Korea', 'Czech Republic'],
    'Group B': ['Canada', 'Bosnia and Herzegovina', 'Qatar', 'Switzerland'],
    'Group C': ['Brazil', 'Morocco', 'Haiti', 'Scotland'],
    'Group D': ['United States', 'Paraguay', 'Australia', 'Turkey'],
    'Group E': ['Germany', 'Curaçao', 'Ivory Coast', 'Ecuador'],
    'Group F': ['Netherlands', 'Japan', 'Sweden', 'Tunisia'],
    'Group G': ['Belgium', 'Egypt', 'Iran', 'New Zealand'],
    'Group H': ['Spain', 'Cape Verde', 'Saudi Arabia', 'Uruguay'],
    'Group I': ['France', 'Senegal', 'Iraq', 'Norway'],
    'Group J': ['Argentina', 'Algeria', 'Austria', 'Jordan'],
    'Group K': ['Portugal', 'DR Congo', 'Uzbekistan', 'Colombia'],
    'Group L': ['England', 'Croatia', 'Ghana', 'Panama'],
}

BRACKET_MATCHUPS = [
    ('1E', '3C'), ('1I', '3D'), ('2A', '2B'), ('1F', '2C'),
    ('2K', '2L'), ('1H', '2J'), ('1D', '3B'), ('1G', '3J'),
    ('1C', '2F'), ('2E', '2I'), ('1A', '3E'), ('1L', '3I'),
    ('1J', '2H'), ('2D', '2G'), ('1B', '3G'), ('1K', '3L'),
]

RESULT_MAP = {'home_win_prob': 'Home Win', 'draw_prob': 'Draw', 'away_win_prob': 'Away Win'}
POINTS_MAP = {'Home Win': 3, 'Draw': 1, 'Away Win': 0}
PROB_COLS = ['home_win_prob', 'draw_prob', 'away_win_prob']

# ---------------------------------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------------------------------

def calculate_match_probs(row):
    """Vectorised Poisson probability calculation for a single match row."""
    goals = np.arange(MAX_GOALS)
    home_probs = poisson.pmf(goals, row['home_xg'])
    away_probs = poisson.pmf(goals, row['away_xg'])

    # Outer product gives all score-combination probabilities
    score_matrix = np.outer(home_probs, away_probs)

    home_win = np.tril(score_matrix, -1).sum()   # home_goals > away_goals
    away_win = np.triu(score_matrix, 1).sum()    # away_goals > home_goals
    draw     = np.trace(score_matrix)

    return pd.Series({'home_win_prob': home_win, 'draw_prob': draw, 'away_win_prob': away_win})


def converting_data(dataset):
    """Reshape a match dataset into one row per team per match."""
    return pd.concat([
        dataset[['home_team', 'away_team', 'home_score', 'neutral']]
            .rename(columns={'home_team': 'team', 'away_team': 'opponent', 'home_score': 'goals'})
            .assign(home=1),
        dataset[['away_team', 'home_team', 'away_score', 'neutral']]
            .rename(columns={'away_team': 'team', 'home_team': 'opponent', 'away_score': 'goals'})
            .assign(home=0),
    ])


def predict_and_finalize(match_df, model, csv_filename):
    """
    Given a DataFrame of matchups (columns: match_id, team, opponent, home),
    run the Poisson model, compute probabilities, label results, assign points,
    save to CSV, and return (final_predictions_df, standings_df, winners_df).
    """
    predict_cols = ['team', 'opponent', 'home']

    home_input = match_df[predict_cols].assign(home=1)
    away_input = match_df[predict_cols].assign(
        team=match_df['opponent'].values,
        opponent=match_df['team'].values,
        home=0
    )

    df = match_df.copy()
    df['home_xg'] = model.predict(home_input)
    df['away_xg']  = model.predict(away_input)

    # Probs
    probs = df.apply(calculate_match_probs, axis=1)
    df = pd.concat([df, probs], axis=1)

    df['predicted_result'] = df[PROB_COLS].idxmax(axis=1).map(RESULT_MAP)

    # Points
    df['match_points'] = df['predicted_result'].map(POINTS_MAP).fillna(0).astype(int)

    for col in PROB_COLS:
        df[col] = (df[col] * 100).round(1).astype(str) + '%'

    save_cols = [c for c in ['match_id', 'team', 'opponent', 'home_xg', 'away_xg',
                              *PROB_COLS, 'predicted_result', 'match_points'] if c in df.columns]
    df[save_cols].to_csv(csv_filename, index=False)

    # Standings
    group_col = 'match_id' if 'match_id' in df.columns else None
    agg_by = [group_col, 'team'] if group_col else ['team']
    standings = df.groupby(agg_by).agg(
        Total_Points=('match_points', 'sum'),
        Total_xG=('home_xg', 'sum')
    ).reset_index()

    sort_by = ([group_col, 'Total_Points', 'Total_xG'] if group_col
               else ['Total_Points', 'Total_xG'])
    standings = standings.sort_values(by=sort_by, ascending=[True, False, False])

    winners = (standings.drop_duplicates(subset=[group_col], keep='first').copy()
               if group_col else standings.head(1).copy())
    winners = winners.reset_index(drop=True)

    return df, standings, winners


def build_knockout_bracket(advancing_teams_df):
    """
    Pair consecutive winners (row 0 vs 1, row 2 vs 3, …) into a new round's
    matchup DataFrame. Returns a DataFrame with columns:
    match_id, team, opponent, home.
    """
    df = advancing_teams_df.copy()
    df['match_num'] = df['match_id'].str.extract(r'(\d+)').astype(int)
    df = df.sort_values('match_num').reset_index(drop=True)

    rows = []
    for match_num, (i) in enumerate(range(0, len(df), 2), start=1):
        team_a = df.iloc[i]['team']
        team_b = df.iloc[i + 1]['team']
        rows += [
            {'match_id': f"Match {match_num}", 'team': team_a, 'opponent': team_b, 'home': 1},
            {'match_id': f"Match {match_num}", 'team': team_b, 'opponent': team_a, 'home': 0},
        ]
    return pd.DataFrame(rows)


def fetch_knockout_team(code, standings_df, third_places_df):
    """Return team name for a bracket code like '1A', '2K', or '3E'."""
    position  = int(code[0])
    group_name = f"Group {code[1]}"
    if position in (1, 2):
        return standings_df.loc[
            (standings_df['group'] == group_name) & (standings_df['Position'] == position),
            'team'
        ].values[0]
    return third_places_df.loc[third_places_df['group'] == group_name, 'team'].values[0]


# ---------------------------------------------------------------------------
# DATA LOADING & PREPARATION
# ---------------------------------------------------------------------------

df = pd.read_csv('results.csv')
df['home_score'] = pd.to_numeric(df['home_score'], errors='coerce')
df['away_score'] = pd.to_numeric(df['away_score'], errors='coerce')

is_wc = df['tournament'] == 'FIFA World Cup'
world_cup_2026_data = df[is_wc & df['home_score'].isna() & df['away_score'].isna()]
df = df[~is_wc & df['home_score'].notna() & df['away_score'].notna()]

# Training data — keep only teams with ≥ 20 appearances
train_historic_data = converting_data(df)
established_teams   = train_historic_data['team'].value_counts()
train_historic_data = train_historic_data[
    train_historic_data['team'].isin(established_teams[established_teams >= 20].index)
]

# ---------------------------------------------------------------------------
# MODEL TRAINING
# ---------------------------------------------------------------------------

poisson_model = smf.glm(
    formula="goals ~ team + opponent + home",
    data=train_historic_data,
    family=sm.families.Poisson()
).fit()

print("Model trained successfully!")
print(poisson_model.summary())

# ---------------------------------------------------------------------------
# GROUP STAGE
# ---------------------------------------------------------------------------

test_world_cup_data = converting_data(world_cup_2026_data)

# Build a reverse lookup: team → group
team_to_group = {team: grp for grp, teams in GROUP_ASSIGNMENTS.items() for team in teams}
test_world_cup_data['group'] = test_world_cup_data['team'].map(team_to_group).fillna('Unassigned')

predict_input = test_world_cup_data.copy()
predict_input['match_id'] = predict_input['group']  

home_input = predict_input[['team', 'opponent']].assign(home=1)
away_input = predict_input[['team', 'opponent']].assign(
    team=predict_input['opponent'].values,
    opponent=predict_input['team'].values,
    home=0
)
predict_input['home_xg'] = poisson_model.predict(home_input)
predict_input['away_xg'] = poisson_model.predict(away_input)

probs = predict_input.apply(calculate_match_probs, axis=1)
final_predictions = pd.concat([predict_input, probs], axis=1)
final_predictions['predicted_result'] = final_predictions[PROB_COLS].idxmax(axis=1).map(RESULT_MAP)
final_predictions['match_points']     = final_predictions['predicted_result'].map(POINTS_MAP).fillna(0).astype(int)

for col in PROB_COLS:
    final_predictions[col] = (final_predictions[col] * 100).round(1).astype(str) + '%'

display_cols = ['team', 'opponent', 'group', 'home_xg', 'away_xg', *PROB_COLS, 'predicted_result', 'match_points']
print(final_predictions[display_cols])

# Group-stage standings
standings = final_predictions.groupby(['group', 'team']).agg(
    Total_Points=('match_points', 'sum'),
    Total_xG=('home_xg', 'sum')
).reset_index()
standings = standings.sort_values(['group', 'Total_Points', 'Total_xG'], ascending=[True, False, False])
standings['Position'] = standings.groupby('group').cumcount() + 1

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
print(standings)

# Export
save_cols = ['team', 'opponent', 'group', 'home_xg', 'away_xg', *PROB_COLS, 'predicted_result', 'match_points']
final_predictions[save_cols].to_csv('predictions.csv', index=False)
print("Data successfully exported to predictions.csv!")

standings[['group', 'Position', 'team', 'Total_Points', 'Total_xG']].to_csv('standings.csv', index=False)
print("Standings exported!")
print(standings[['group', 'Position', 'team', 'Total_Points', 'Total_xG']].to_string(index=False))

# ---------------------------------------------------------------------------
# THIRD-PLACE RANKINGS
# ---------------------------------------------------------------------------

third_places = (
    standings[standings['Position'] == 3]
    .copy()
    .sort_values(['Total_Points', 'Total_xG'], ascending=False)
    .reset_index(drop=True)
)
third_places['Overall_Rank'] = third_places.index + 1
third_places['Qualification'] = np.where(
    third_places['Overall_Rank'] <= 8, 'Advance to knockout stage', 'Eliminated'
)
third_places = third_places[['Overall_Rank', 'group', 'team', 'Total_Points', 'Total_xG', 'Qualification']]
third_places.to_csv('third_placed_rankings.csv', index=False)
print("--- Ranking of Third-Placed Teams ---")
print(third_places.to_string(index=False))

# ---------------------------------------------------------------------------
# KNOCKOUT ROUNDS
# ---------------------------------------------------------------------------

# --- Round of 32 (initial bracket from group-stage results) ---
knockout_rows = []
for match_num, (home_code, away_code) in enumerate(BRACKET_MATCHUPS, start=1):
    team_a = fetch_knockout_team(home_code, standings, third_places)
    team_b = fetch_knockout_team(away_code, standings, third_places)
    knockout_rows += [
        {'match_id': f"Match {match_num}", 'team': team_a, 'opponent': team_b, 'home': 1},
        {'match_id': f"Match {match_num}", 'team': team_b, 'opponent': team_a, 'home': 0},
    ]

current_bracket = pd.DataFrame(knockout_rows)

round_configs = [
    ('predictions_32.csv', 'Round of 32', 16),
    ('predictions_16.csv', 'Round of 16',  8),
    ('predictions_8.csv',  'Round of 8',   4),
    ('predictions_4.csv',  'Round of 4',   2),
    ('predictions_2.csv',  'Final',        1),
]

for csv_file, round_name, expected_winners in round_configs:
    _, standings_round, winners = predict_and_finalize(current_bracket, poisson_model, csv_file)
    print(f"\n--- {round_name} Standings ---")
    print(standings_round)
    print(f"\n--- {round_name} Advancing Teams ---")
    print(winners)

    if expected_winners == 1:
        print("\n--- TOURNAMENT WINNER ---")
        print(winners.iloc[0]['team'])
        break

    current_bracket = build_knockout_bracket(winners)