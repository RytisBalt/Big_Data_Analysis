import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import poisson

def calculate_match_probs(row, max_goals=8):
    home_xg = row['home_xg']
    away_xg = row['away_xg']
    
    home_goal_probs = [poisson.pmf(i, home_xg) for i in range(max_goals)]
    away_goal_probs = [poisson.pmf(i, away_xg) for i in range(max_goals)]
    
    home_win_prob = 0
    draw_prob = 0
    away_win_prob = 0
    
    for home_goals in range(max_goals):
        for away_goals in range(max_goals):
            score_prob = home_goal_probs[home_goals] * away_goal_probs[away_goals]
            
            if home_goals > away_goals:
                home_win_prob += score_prob
            elif home_goals < away_goals:
                away_win_prob += score_prob
            else:
                draw_prob += score_prob
                
    return pd.Series({
        'home_win_prob': home_win_prob,
        'draw_prob': draw_prob,
        'away_win_prob': away_win_prob
    })


def converting_data(dataset):
    data = pd.concat([
    dataset[['home_team', 'away_team', 'home_score', 'neutral']].rename(
        columns={'home_team': 'team', 'away_team': 'opponent', 'home_score': 'goals'}
    ).assign(home=1), 
    
    dataset[['away_team', 'home_team', 'away_score', 'neutral']].rename(
        columns={'away_team': 'team', 'home_team': 'opponent', 'away_score': 'goals'}
    ).assign(home=0)
])
    return data






df = pd.read_csv('results.csv')

df['home_score'] = pd.to_numeric(df['home_score'], errors='coerce')
df['away_score'] = pd.to_numeric(df['away_score'], errors='coerce')

world_cup_2026_data = df[
    (df['tournament'] == 'FIFA World Cup') & 
    (df['home_score'].isna()) & 
    (df['away_score'].isna())
]

df = df[
    (df['tournament'] != 'FIFA World Cup') & 
    (df['home_score'].notna()) & 
    (df['away_score'].notna())
]
train_historic_data = converting_data(df)
team_counts = train_historic_data['team'].value_counts()
established_teams = team_counts[team_counts >= 20].index
train_historic_data = train_historic_data[
    (train_historic_data['team'].isin(established_teams))
]
test_world_cup_data = converting_data(world_cup_2026_data)

conditions = [
    test_world_cup_data['team'].isin(['Mexico', 'South Africa', 'South Korea', 'Czech Republic']),
    test_world_cup_data['team'].isin(['Canada', 'Bosnia and Herzegovina', 'Qatar', 'Switzerland']),
    test_world_cup_data['team'].isin(['Brazil', 'Morocco', 'Haiti', 'Scotland']),
    test_world_cup_data['team'].isin(['United States', 'Paraguay', 'Australia', 'Turkey']),
    test_world_cup_data['team'].isin(['Germany', 'Curaçao', 'Ivory Coast', 'Ecuador']),
    test_world_cup_data['team'].isin(['Netherlands', 'Japan', 'Sweden', 'Tunisia']),
    test_world_cup_data['team'].isin(['Belgium', 'Egypt', 'Iran', 'New Zealand']),
    test_world_cup_data['team'].isin(['Spain', 'Cape Verde', 'Saudi Arabia', 'Uruguay']),
    test_world_cup_data['team'].isin(['France', 'Senegal', 'Iraq', 'Norway']),
    test_world_cup_data['team'].isin(['Argentina', 'Algeria', 'Austria', 'Jordan']),
    test_world_cup_data['team'].isin(['Portugal', 'DR Congo', 'Uzbekistan', 'Colombia']),
    test_world_cup_data['team'].isin(['England', 'Croatia', 'Ghana', 'Panama'])
]

choices = [
    'Group A',
    'Group B',
    'Group C',
    'Group D',
    'Group E',
    'Group F',
    'Group G',
    'Group H',
    'Group I',
    'Group J',
    'Group K',
    'Group L'
]

poisson_model = smf.glm(formula="goals ~ team + opponent + home", 
                        data=train_historic_data, 
                        family=sm.families.Poisson()).fit()

print("Model trained successfully!")
print(poisson_model.summary())


home_attack = pd.DataFrame({
    'team': test_world_cup_data['team'], 
    'opponent': test_world_cup_data['opponent'], 
    'home': 1,
})

away_attack = pd.DataFrame({
    'team': test_world_cup_data['opponent'], 
    'opponent': test_world_cup_data['team'], 
    'home': 0,
})

test_world_cup_data['home_xg'] = poisson_model.predict(home_attack)
test_world_cup_data['away_xg'] = poisson_model.predict(away_attack)
probabilities = test_world_cup_data.apply(calculate_match_probs, axis=1)
final_predictions = pd.concat([test_world_cup_data, probabilities], axis=1)

prob_cols = ['home_win_prob', 'draw_prob', 'away_win_prob']
final_predictions['predicted_result'] = final_predictions[prob_cols].idxmax(axis=1)

final_predictions['predicted_result'] = final_predictions['predicted_result'].map({
    'home_win_prob': 'Home Win',
    'draw_prob': 'Draw',
    'away_win_prob': 'Away Win'
})

final_predictions['home_win_prob'] = (final_predictions['home_win_prob'] * 100).round(1).astype(str) + '%'
final_predictions['draw_prob'] = (final_predictions['draw_prob'] * 100).round(1).astype(str) + '%'
final_predictions['away_win_prob'] = (final_predictions['away_win_prob'] * 100).round(1).astype(str) + '%'
final_predictions['group'] = np.select(conditions, choices, default='Unassigned')

possible_results = [
    final_predictions['predicted_result'] == 'Home Win',
    final_predictions['predicted_result'] == 'Draw',
    final_predictions['predicted_result'] == 'Away Win'
]
choices_results = [3, 1, 0]

final_predictions['match_points'] = np.select(possible_results, choices_results, default=0)
columns_to_save = ['team', 'opponent', 'group', 'home_xg', 'away_xg', 'home_win_prob', 'draw_prob', 'away_win_prob', 'predicted_result', 'match_points']
print(final_predictions[['team', 'opponent', 'group', 'home_xg', 'away_xg', 'home_win_prob', 'draw_prob', 'away_win_prob', 'predicted_result', 'match_points']])

standings = final_predictions.groupby(['group', 'team']).agg(
    Total_Points=('match_points', 'sum'),
    Total_xG=('home_xg', 'sum')
).reset_index()

standings = standings.sort_values(
    by=['group', 'Total_Points', 'Total_xG'], 
    ascending=[True, False, False]
)

standings['Position'] = standings.groupby('group').cumcount() + 1

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
print(standings)
final_predictions[columns_to_save].to_csv('predictions.csv', index=False)
print("Data successfully exported to predictions.csv!")

standings = standings[['group', 'Position', 'team', 'Total_Points', 'Total_xG']]
print(standings.to_string(index=False))

standings_columns_to_save = ['group', 'team', 'Total_Points', 'Total_xG', 'Position']
standings[standings_columns_to_save].to_csv('standings.csv', index = False)
print("Standings exported!")


### THIRD PLACES MANIPULATION ###

third_places = standings[standings['Position'] == 3].copy()
third_places = third_places.sort_values(
    by=['Total_Points', 'Total_xG'], 
    ascending=[False, False]
)

third_places = third_places.reset_index(drop=True)
third_places['Overall_Rank'] = third_places.index + 1

third_places['Qualification'] = np.where(
    third_places['Overall_Rank'] <= 8, 
    'Advance to knockout stage', 
    'Eliminated'
)

third_places = third_places[['Overall_Rank', 'group', 'team', 'Total_Points', 'Total_xG', 'Qualification']]

third_places.to_csv('third_placed_rankings.csv', index=False)

print("--- Ranking of Third-Placed Teams ---")
print(third_places.to_string(index=False))

### KNOCKOUT STAGE ###

bracket_matchups = [
    ('1E', '3C'), # Match 1: Winner Group E vs 3rd Place
    ('1I', '3D'), # Match 2: Winner Group I vs 3rd Place
    ('2A', '2B'), # Match 3: Runner-up Group A vs Runner-up Group B
    ('1F', '2C'), # Match 4: Winner Group F vs Runner-up Group C
    ('2K', '2L'), # Match 5: Runner-up Group K vs Runner-up Group L
    ('1H', '2J'), # Match 6: Winner Group H vs Runner-up Group J
    ('1D', '3B'), # Match 7: Winner Group D vs 3rd Place
    ('1G', '3J'), # Match 8: Winner Group G vs 3rd Place
    ('1C', '2F'), # Match 9: Winner Group C vs Runner-up Group F
    ('2E', '2I'), # Match 10: Runner-up Group E vs Runner-up Group I
    ('1A', '3E'), # Match 11: Winner Group A vs 3rd Place
    ('1L', '3I'), # Match 12: Winner Group L vs 3rd Place
    ('1J', '2H'), # Match 13: Winner Group J vs Runner-up Group H
    ('2D', '2G'), # Match 14: Runner-up Group D vs Runner-up Group G
    ('1B', '3G'), # Match 15: Winner Group B vs 3rd Place
    ('1K', '3L')  # Match 16: Winner Group K vs 3rd Place
]

def fetch_knockout_team(code, standings_df, third_places_df):
    """Automatically finds the country name based on a code like '1A' or '3E'"""
    position = int(code[0])
    group_name = f"Group {code[1]}"
    
    if position in [1, 2]:
        team = standings_df[
            (standings_df['group'] == group_name) & 
            (standings_df['Position'] == position)
        ]['team'].values[0]
    else:
        team = third_places_df[
            third_places_df['group'] == group_name
        ]['team'].values[0]
        
    return team

knockout_rows = []
match_number = 1

for home_code, away_code in bracket_matchups:
    team_a = fetch_knockout_team(home_code, standings, third_places)
    team_b = fetch_knockout_team(away_code, standings, third_places)
    
    knockout_rows.append({
        'match_id': f"Match {match_number}",
        'team': team_a, 
        'opponent': team_b, 
        'home': 1 
    })
    
    knockout_rows.append({
        'match_id': f"Match {match_number}",
        'team': team_b, 
        'opponent': team_a, 
        'home': 0
    })
    
    match_number += 1

round_of_32_df = pd.DataFrame(knockout_rows)

home_attack_r32 = pd.DataFrame({
    'team': round_of_32_df['team'], 
    'opponent': round_of_32_df['opponent'], 
    'home': 1,
    'match_id' : round_of_32_df['match_id'] 
})

away_attack_r32 = pd.DataFrame({
    'team': round_of_32_df['opponent'], 
    'opponent': round_of_32_df['team'], 
    'home': 0,
    'match_id': round_of_32_df['match_id']
})

round_of_32_df['home_xg'] = poisson_model.predict(home_attack_r32)
round_of_32_df['away_xg'] = poisson_model.predict(away_attack_r32)
probabilities_r32 = round_of_32_df.apply(calculate_match_probs, axis=1)
final_predictions_r32 = pd.concat([round_of_32_df, probabilities_r32], axis=1)
prob_cols = ['home_win_prob', 'draw_prob', 'away_win_prob']
final_predictions_r32['predicted_result'] = final_predictions_r32[prob_cols].idxmax(axis=1)

final_predictions_r32['predicted_result'] = final_predictions_r32['predicted_result'].map({
    'home_win_prob': 'Home Win',
    'draw_prob': 'Draw',
    'away_win_prob': 'Away Win'
})

final_predictions_r32['home_win_prob'] = (final_predictions_r32['home_win_prob'] * 100).round(1).astype(str) + '%'
final_predictions_r32['draw_prob'] = (final_predictions_r32['draw_prob'] * 100).round(1).astype(str) + '%'
final_predictions_r32['away_win_prob'] = (final_predictions_r32['away_win_prob'] * 100).round(1).astype(str) + '%'
possible_results = [
    final_predictions_r32['predicted_result'] == 'Home Win',
    final_predictions_r32['predicted_result'] == 'Draw',
    final_predictions_r32['predicted_result'] == 'Away Win'
]
choices_results = [3, 1, 0]
final_predictions_r32['match_points'] = np.select(possible_results, choices_results, default=0)
columns_to_save = ['match_id', 'team', 'opponent', 'home_xg', 'away_xg', 'home_win_prob', 'draw_prob', 'away_win_prob', 'predicted_result', 'match_points']
final_predictions_r32[columns_to_save].to_csv('predictions_32.csv', index=False)

standings_after_r32 = final_predictions_r32.groupby(['match_id', 'team']).agg(
    Total_Points=('match_points', 'sum'),
    Total_xG=('home_xg', 'sum')
).reset_index()

standings_after_r32 = standings_after_r32.sort_values(
    by=['match_id', 'Total_Points', 'Total_xG'], 
    ascending=[True, False, False]
)
print(standings_after_r32)
round_of_16_teams = standings_after_r32.drop_duplicates(subset=['match_id'], keep='first').copy()
round_of_16_teams = round_of_16_teams.reset_index(drop=True)

print("--- The 16 Advancing Teams ---")
print(round_of_16_teams)

round_of_16_teams['match_num'] = round_of_16_teams['match_id'].str.extract(r'(\d+)').astype(int)
round_of_16_teams = round_of_16_teams.sort_values(by='match_num', ascending=True).reset_index(drop=True)

print("--- Verifying Sorted Order ---")
print(round_of_16_teams[['match_num', 'team']].head(4))
print("-" * 30)

# 2. BUILD THE BRACKET (Match 1 vs 2, Match 3 vs 4)
knockout_rows = []
r16_match_num = 1

for i in range(0, len(round_of_16_teams), 2):
    team_a = round_of_16_teams.iloc[i]['team']       # Match 1 Winner (Germany)
    team_b = round_of_16_teams.iloc[i+1]['team']     # Match 2 Winner (France)
    
    knockout_rows.append({
        'match_id': f"Match {r16_match_num}",
        'team': team_a,
        'opponent': team_b,
        'home': 1
    })
    
    knockout_rows.append({
        'match_id': f"Match {r16_match_num}",
        'team': team_b,
        'opponent': team_a,
        'home': 0
    })
    
    r16_match_num += 1

# 3. CREATE THE DATAFRAME
round_of_16_df = pd.DataFrame(knockout_rows)

print("--- Round of 16 Matchups ---")

home_attack_r16 = pd.DataFrame({
    'team': round_of_16_df['team'], 
    'opponent': round_of_16_df['opponent'], 
    'home': 1,
    'match_id' : round_of_16_df['match_id'] 
})

away_attack_r16 = pd.DataFrame({
    'team': round_of_16_df['opponent'], 
    'opponent': round_of_16_df['team'], 
    'home': 0,
    'match_id': round_of_16_df['match_id']
})
round_of_16_df['home_xg'] = poisson_model.predict(home_attack_r16)
round_of_16_df['away_xg'] = poisson_model.predict(away_attack_r16)
probabilities_r16 = round_of_16_df.apply(calculate_match_probs, axis=1)
final_predictions_r16 = pd.concat([round_of_16_df, probabilities_r16], axis=1)
prob_cols = ['home_win_prob', 'draw_prob', 'away_win_prob']
final_predictions_r16['predicted_result'] = final_predictions_r16[prob_cols].idxmax(axis=1)

final_predictions_r16['predicted_result'] = final_predictions_r16['predicted_result'].map({
    'home_win_prob': 'Home Win',
    'draw_prob': 'Draw',
    'away_win_prob': 'Away Win'
})

final_predictions_r16['home_win_prob'] = (final_predictions_r16['home_win_prob'] * 100).round(1).astype(str) + '%'
final_predictions_r16['draw_prob'] = (final_predictions_r16['draw_prob'] * 100).round(1).astype(str) + '%'
final_predictions_r16['away_win_prob'] = (final_predictions_r16['away_win_prob'] * 100).round(1).astype(str) + '%'
possible_results = [
    final_predictions_r16['predicted_result'] == 'Home Win',
    final_predictions_r16['predicted_result'] == 'Draw',
    final_predictions_r16['predicted_result'] == 'Away Win'
]
choices_results = [3, 1, 0]
final_predictions_r16['match_points'] = np.select(possible_results, choices_results, default=0)
columns_to_save = ['match_id', 'team', 'opponent', 'home_xg', 'away_xg', 'home_win_prob', 'draw_prob', 'away_win_prob', 'predicted_result', 'match_points']
final_predictions_r16[columns_to_save].to_csv('predictions_16.csv', index=False)

standings_after_r16 = final_predictions_r16.groupby(['match_id', 'team']).agg(
    Total_Points=('match_points', 'sum'),
    Total_xG=('home_xg', 'sum')
).reset_index()

standings_after_r16 = standings_after_r16.sort_values(
    by=['match_id', 'Total_Points', 'Total_xG'], 
    ascending=[True, False, False]
)
print(standings_after_r16)
round_of_8_teams = standings_after_r16.drop_duplicates(subset=['match_id'], keep='first').copy()
round_of_8_teams = round_of_8_teams.reset_index(drop=True)

print("--- Top 8 Advancing Teams ---")
print(round_of_8_teams)







round_of_8_teams['match_num'] = round_of_8_teams['match_id'].str.extract(r'(\d+)').astype(int)
round_of_8_teams = round_of_8_teams.sort_values(by='match_num', ascending=True).reset_index(drop=True)

print("--- Verifying Sorted Order ---")
print(round_of_8_teams[['match_num', 'team']].head(4))
print("-" * 30)

# 2. BUILD THE BRACKET (Match 1 vs 2, Match 3 vs 4)
knockout_rows = []
r8_match_num = 1

for i in range(0, len(round_of_8_teams), 2):
    team_a = round_of_8_teams.iloc[i]['team']       # Match 1 Winner (Germany)
    team_b = round_of_8_teams.iloc[i+1]['team']     # Match 2 Winner (France)
    
    knockout_rows.append({
        'match_id': f"Match {r8_match_num}",
        'team': team_a,
        'opponent': team_b,
        'home': 1
    })
    
    knockout_rows.append({
        'match_id': f"Match {r8_match_num}",
        'team': team_b,
        'opponent': team_a,
        'home': 0
    })
    
    r8_match_num += 1

round_of_8_df = pd.DataFrame(knockout_rows)

print("--- Round of 8 Matchups ---")

home_attack_r8 = pd.DataFrame({
    'team': round_of_8_df['team'], 
    'opponent': round_of_8_df['opponent'], 
    'home': 1,
    'match_id' : round_of_8_df['match_id'] 
})

away_attack_r8 = pd.DataFrame({
    'team': round_of_8_df['opponent'], 
    'opponent': round_of_8_df['team'], 
    'home': 0,
    'match_id': round_of_8_df['match_id']
})
round_of_8_df['home_xg'] = poisson_model.predict(home_attack_r8)
round_of_8_df['away_xg'] = poisson_model.predict(away_attack_r8)
probabilities_r8 = round_of_8_df.apply(calculate_match_probs, axis=1)
final_predictions_r8 = pd.concat([round_of_8_df, probabilities_r8], axis=1)
prob_cols = ['home_win_prob', 'draw_prob', 'away_win_prob']
final_predictions_r8['predicted_result'] = final_predictions_r8[prob_cols].idxmax(axis=1)

final_predictions_r8['predicted_result'] = final_predictions_r8['predicted_result'].map({
    'home_win_prob': 'Home Win',
    'draw_prob': 'Draw',
    'away_win_prob': 'Away Win'
})

final_predictions_r8['home_win_prob'] = (final_predictions_r8['home_win_prob'] * 100).round(1).astype(str) + '%'
final_predictions_r8['draw_prob'] = (final_predictions_r8['draw_prob'] * 100).round(1).astype(str) + '%'
final_predictions_r8['away_win_prob'] = (final_predictions_r8['away_win_prob'] * 100).round(1).astype(str) + '%'
possible_results = [
    final_predictions_r8['predicted_result'] == 'Home Win',
    final_predictions_r8['predicted_result'] == 'Draw',
    final_predictions_r8['predicted_result'] == 'Away Win'
]
choices_results = [3, 1, 0]
final_predictions_r8['match_points'] = np.select(possible_results, choices_results, default=0)
columns_to_save = ['match_id', 'team', 'opponent', 'home_xg', 'away_xg', 'home_win_prob', 'draw_prob', 'away_win_prob', 'predicted_result', 'match_points']
final_predictions_r8[columns_to_save].to_csv('predictions_8.csv', index=False)

standings_after_r8 = final_predictions_r8.groupby(['match_id', 'team']).agg(
    Total_Points=('match_points', 'sum'),
    Total_xG=('home_xg', 'sum')
).reset_index()

standings_after_r8 = standings_after_r8.sort_values(
    by=['match_id', 'Total_Points', 'Total_xG'], 
    ascending=[True, False, False]
)
print(standings_after_r8)
round_of_4_teams = standings_after_r8.drop_duplicates(subset=['match_id'], keep='first').copy()
round_of_4_teams = round_of_4_teams.reset_index(drop=True)

print("--- Top 4 Advancing Teams ---")
print(round_of_4_teams)











round_of_4_teams['match_num'] = round_of_4_teams['match_id'].str.extract(r'(\d+)').astype(int)
round_of_4_teams = round_of_4_teams.sort_values(by='match_num', ascending=True).reset_index(drop=True)

print("--- Verifying Sorted Order ---")
print(round_of_4_teams[['match_num', 'team']].head(4))
print("-" * 30)

# 2. BUILD THE BRACKET (Match 1 vs 2, Match 3 vs 4)
knockout_rows = []
r4_match_num = 1

for i in range(0, len(round_of_4_teams), 2):
    team_a = round_of_4_teams.iloc[i]['team']       # Match 1 Winner (Germany)
    team_b = round_of_4_teams.iloc[i+1]['team']     # Match 2 Winner (France)
    
    knockout_rows.append({
        'match_id': f"Match {r4_match_num}",
        'team': team_a,
        'opponent': team_b,
        'home': 1
    })
    
    knockout_rows.append({
        'match_id': f"Match {r4_match_num}",
        'team': team_b,
        'opponent': team_a,
        'home': 0
    })
    
    r4_match_num += 1

round_of_4_df = pd.DataFrame(knockout_rows)

print("--- Round of 8 Matchups ---")

home_attack_r4 = pd.DataFrame({
    'team': round_of_4_df['team'], 
    'opponent': round_of_4_df['opponent'], 
    'home': 1,
    'match_id' : round_of_4_df['match_id'] 
})

away_attack_r4 = pd.DataFrame({
    'team': round_of_4_df['opponent'], 
    'opponent': round_of_4_df['team'], 
    'home': 0,
    'match_id': round_of_4_df['match_id']
})
round_of_4_df['home_xg'] = poisson_model.predict(home_attack_r4)
round_of_4_df['away_xg'] = poisson_model.predict(away_attack_r4)
probabilities_r4 = round_of_4_df.apply(calculate_match_probs, axis=1)
final_predictions_r4 = pd.concat([round_of_4_df, probabilities_r4], axis=1)
prob_cols = ['home_win_prob', 'draw_prob', 'away_win_prob']
final_predictions_r4['predicted_result'] = final_predictions_r4[prob_cols].idxmax(axis=1)

final_predictions_r4['predicted_result'] = final_predictions_r4['predicted_result'].map({
    'home_win_prob': 'Home Win',
    'draw_prob': 'Draw',
    'away_win_prob': 'Away Win'
})

final_predictions_r4['home_win_prob'] = (final_predictions_r4['home_win_prob'] * 100).round(1).astype(str) + '%'
final_predictions_r4['draw_prob'] = (final_predictions_r4['draw_prob'] * 100).round(1).astype(str) + '%'
final_predictions_r4['away_win_prob'] = (final_predictions_r4['away_win_prob'] * 100).round(1).astype(str) + '%'
possible_results = [
    final_predictions_r4['predicted_result'] == 'Home Win',
    final_predictions_r4['predicted_result'] == 'Draw',
    final_predictions_r4['predicted_result'] == 'Away Win'
]
choices_results = [3, 1, 0]
final_predictions_r4['match_points'] = np.select(possible_results, choices_results, default=0)
columns_to_save = ['match_id', 'team', 'opponent', 'home_xg', 'away_xg', 'home_win_prob', 'draw_prob', 'away_win_prob', 'predicted_result', 'match_points']
final_predictions_r4[columns_to_save].to_csv('predictions_8.csv', index=False)

standings_after_r4 = final_predictions_r4.groupby(['match_id', 'team']).agg(
    Total_Points=('match_points', 'sum'),
    Total_xG=('home_xg', 'sum')
).reset_index()

standings_after_r4 = standings_after_r4.sort_values(
    by=['match_id', 'Total_Points', 'Total_xG'], 
    ascending=[True, False, False]
)
print(standings_after_r4)
round_of_2_teams = standings_after_r4.drop_duplicates(subset=['match_id'], keep='first').copy()
round_of_2_teams = round_of_2_teams.reset_index(drop=True)

print("--- Top 2 Advancing Teams ---")
print(round_of_2_teams)







round_of_2_teams['match_num'] = round_of_2_teams['match_id'].str.extract(r'(\d+)').astype(int)
round_of_2_teams = round_of_2_teams.sort_values(by='match_num', ascending=True).reset_index(drop=True)

print("--- Verifying Sorted Order ---")
print(round_of_2_teams[['match_num', 'team']].head(4))
print("-" * 30)

# 2. BUILD THE BRACKET (Match 1 vs 2, Match 3 vs 4)
knockout_rows = []
r2_match_num = 1

for i in range(0, len(round_of_2_teams), 2):
    team_a = round_of_2_teams.iloc[i]['team']       # Match 1 Winner (Germany)
    team_b = round_of_2_teams.iloc[i+1]['team']     # Match 2 Winner (France)
    
    knockout_rows.append({
        'match_id': f"Match {r2_match_num}",
        'team': team_a,
        'opponent': team_b,
        'home': 1
    })
    
    knockout_rows.append({
        'match_id': f"Match {r2_match_num}",
        'team': team_b,
        'opponent': team_a,
        'home': 0
    })
    
    r2_match_num += 1

round_of_2_df = pd.DataFrame(knockout_rows)

print("--- Round of 8 Matchups ---")

home_attack_r2 = pd.DataFrame({
    'team': round_of_2_df['team'], 
    'opponent': round_of_2_df['opponent'], 
    'home': 1,
    'match_id' : round_of_2_df['match_id'] 
})

away_attack_r2 = pd.DataFrame({
    'team': round_of_2_df['opponent'], 
    'opponent': round_of_2_df['team'], 
    'home': 0,
    'match_id': round_of_2_df['match_id']
})
round_of_2_df['home_xg'] = poisson_model.predict(home_attack_r2)
round_of_2_df['away_xg'] = poisson_model.predict(away_attack_r2)
probabilities_r2 = round_of_2_df.apply(calculate_match_probs, axis=1)
final_predictions_r2 = pd.concat([round_of_2_df, probabilities_r2], axis=1)
prob_cols = ['home_win_prob', 'draw_prob', 'away_win_prob']
final_predictions_r2['predicted_result'] = final_predictions_r2[prob_cols].idxmax(axis=1)

final_predictions_r2['predicted_result'] = final_predictions_r2['predicted_result'].map({
    'home_win_prob': 'Home Win',
    'draw_prob': 'Draw',
    'away_win_prob': 'Away Win'
})

final_predictions_r2['home_win_prob'] = (final_predictions_r2['home_win_prob'] * 100).round(1).astype(str) + '%'
final_predictions_r2['draw_prob'] = (final_predictions_r2['draw_prob'] * 100).round(1).astype(str) + '%'
final_predictions_r2['away_win_prob'] = (final_predictions_r2['away_win_prob'] * 100).round(1).astype(str) + '%'
possible_results = [
    final_predictions_r2['predicted_result'] == 'Home Win',
    final_predictions_r2['predicted_result'] == 'Draw',
    final_predictions_r2['predicted_result'] == 'Away Win'
]
choices_results = [3, 1, 0]
final_predictions_r2['match_points'] = np.select(possible_results, choices_results, default=0)
columns_to_save = ['match_id', 'team', 'opponent', 'home_xg', 'away_xg', 'home_win_prob', 'draw_prob', 'away_win_prob', 'predicted_result', 'match_points']
final_predictions_r2[columns_to_save].to_csv('predictions_2.csv', index=False)

standings_after_r2 = final_predictions_r2.groupby(['match_id', 'team']).agg(
    Total_Points=('match_points', 'sum'),
    Total_xG=('home_xg', 'sum')
).reset_index()

standings_after_r2 = standings_after_r2.sort_values(
    by=['match_id', 'Total_Points', 'Total_xG'], 
    ascending=[True, False, False]
)
print(standings_after_r2)
winner = standings_after_r2.drop_duplicates(subset=['match_id'], keep='first').copy()
winner = winner.reset_index(drop=True)

print("--- Winner ---")
print(winner)