#=== WORLD CUP DATASET ===#


import pandas as pd
import numpy as np
import xgboost as xgb


hist_matches = pd.read_csv(r"C:\Users\Flavio\OneDrive\Progetti\World Cup\results.csv", sep=';')
hist_matches['date'] = pd.to_datetime(hist_matches['date'])
hist_matches = hist_matches.iloc[:, 0:6]    # .dropna() droppa solo le ultime righe dei quarti

# print(hist_matches)

teams_data_train = pd.read_csv(r'C:\Users\Flavio\OneDrive\Progetti\World Cup\train.csv', sep=';')  # non c'è il market value per il 2002
columns = list(range(0, 9)) + [10, 11, 13]
print(f"DataFrame total columns: {teams_data_train.shape[1]}")
teams_data_train = teams_data_train.iloc[:, columns]

teams_data_test = pd.read_csv(r'C:\Users\Flavio\OneDrive\Progetti\World Cup\test.csv')    # sia nel train che test dataset presi non viene fatto dropna()
# perchè si perdono alcune righe per via di variabili poco importanti che non ci porteremo dietro
columns = list(range(0, 9)) + [10, 11, 13]
teams_data_test = teams_data_test.iloc[:, columns]

# print(teams_data_train)
# print(teams_data_test)

teams_data = pd.concat([teams_data_train, teams_data_test])
teams_data['version'] = pd.to_numeric(teams_data['version'])

# print(teams_data)


world_cup_data = hist_matches[(hist_matches['date'] >= '2004-01-01') & (hist_matches['tournament'] == 'FIFA World Cup')]
world_cup_data['year'] = world_cup_data['date'].dt.year

world_cup_data = pd.merge(world_cup_data, teams_data, left_on=['year', 'home_team'], right_on=['version', 'team'], how='left').dropna().drop(columns=['version', 'team', 'continent']) # con dropna() levo le ultime tre partite e due righe che non so da dove le prende
world_cup_data = world_cup_data.rename(columns={'is_host':'home_host', 'goals_scored_last_4y':'home_scored_4',
       'goals_received_last_4y':'home_received_4', 'wins_last_4y':'home_wins_4', 'losses_last_4y':'home_loss_4',
       'draws_last_4y':'home_draws_4', 'squad_total_market_value_eur':'home_market_value(eur)',
       'fifa_rank_pre_tournament':'home_rank_pretournament', 'squad_avg_age':'home_avg_age'})

world_cup_data = pd.merge(world_cup_data, teams_data, left_on=['year', 'away_team'], right_on=['version', 'team'], how='left').dropna().drop(columns=[ 'version', 'team', 'continent'])
world_cup_data = world_cup_data.rename(columns={'is_host':'away_host', 'goals_scored_last_4y':'away_scored_4',
       'goals_received_last_4y':'away_received_4', 'wins_last_4y':'away_wins_4', 'losses_last_4y':'away_loss_4',
       'draws_last_4y':'away_draws_4', 'squad_total_market_value_eur':'away_market_value(eur)',
       'fifa_rank_pre_tournament':'away_rank_pretournament', 'squad_avg_age':'away_avg_age'})

world_cup_data['scored_4_delta'] = world_cup_data['home_scored_4'] - world_cup_data['away_scored_4']
world_cup_data['received_4_delta'] = world_cup_data['home_received_4'] - world_cup_data['away_received_4']
world_cup_data['wins_4_delta'] = world_cup_data['home_wins_4'] - world_cup_data['away_wins_4']
world_cup_data['loss_4_delta'] = world_cup_data['home_loss_4'] - world_cup_data['away_loss_4']
world_cup_data['draws_4_delta'] = world_cup_data['home_draws_4'] - world_cup_data['away_draws_4']
world_cup_data['market_value_delta(eur)'] = world_cup_data['home_market_value(eur)'] - world_cup_data['away_market_value(eur)']
world_cup_data['rank_pre_tournament_delta'] = world_cup_data['home_rank_pretournament'] - world_cup_data['away_rank_pretournament']
world_cup_data['avg_age_delta'] = world_cup_data['home_avg_age'] - world_cup_data['away_avg_age']

world_cup_data = world_cup_data.drop(columns=['home_scored_4', 'home_received_4', 'home_wins_4', 'home_loss_4',
       'home_draws_4', 'home_market_value(eur)', 'home_rank_pretournament', 'home_avg_age', 'away_scored_4',
       'away_received_4', 'away_wins_4', 'away_loss_4', 'away_draws_4', 'away_market_value(eur)',
       'away_rank_pretournament', 'away_avg_age'])

cond = [world_cup_data['home_score'] > world_cup_data['away_score'], world_cup_data['home_score'] == world_cup_data['away_score']]

val = [0, 1]

world_cup_data['target_col'] = np.select(cond, val, default=2)

# print(world_cup_data.columns)

features = world_cup_data[['year', 'home_host', 'away_host', 'scored_4_delta',
       'received_4_delta', 'wins_4_delta', 'loss_4_delta', 'draws_4_delta',
       'market_value_delta(eur)', 'rank_pre_tournament_delta', 'avg_age_delta']]

target = world_cup_data[['year','target_col']]

# print(world_cup_data)
# print(features)
# print(target)

years = [2006, 2010, 2014, 2018, 2022]
# print(years)

from sklearn.metrics import accuracy_score, log_loss

fold_metrics = []

for y in years:

       X_train = features[(features['year'] != y)].drop(columns=['year'])
       Y_train = target[(target['year'] != y)]['target_col']

       X_test = features[(features['year'] == y)].drop(columns=['year']) # da prendere el colonne
       Y_test = target[(target['year'] == y)]['target_col']

       model = xgb.XGBClassifier(
              objective="multi:softprob", num_class=3,
              max_depth=3, min_child_weight=10,
              subsample=0.7, colsample_bytree=0.7,
              reg_lambda=2.0, n_estimators=300,
              eval_metric="mlogloss", random_state=42,
       )
       model.fit(X_train, Y_train)

       prob = model.predict_proba(X_test)
       preds = model.predict(X_test)

       acc = accuracy_score(Y_test, preds)
       loss = log_loss(Y_test, prob, labels=[0, 1, 2])
    
       fold_metrics.append({
              'Year': y,
              'Accuracy': acc,
              'LogLoss': loss
       })
    
       print(f"Mondiale {y} -> Accuracy: {acc:.4f} | LogLoss: {loss:.4f}")
