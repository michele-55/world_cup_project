#=== WORLD CUP DATASET ===#

import pandas as pd


hist_matches = pd.read_csv(, sep=';')
hist_matches['date'] = pd.to_datetime(hist_matches['date'])
hist_matches = hist_matches.iloc[:, 0:6]    # .dropna() droppa solo le ultime righe dei quarti

# print(hist_matches)

teams_data_train = pd.read_csv(, sep=';')  # non c'è il market value per il 2002
columns = list(range(0, 9)) + [10, 11, 13]
print(f"DataFrame total columns: {teams_data_train.shape[1]}")
teams_data_train = teams_data_train.iloc[:, columns]

teams_data_test = pd.read_csv()    # sia nel train che test dataset presi non viene fatto dropna()
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

world_cup_data = pd.merge(world_cup_data, teams_data, left_on=['year', 'away_team'], right_on=['version', 'team'], how='left').dropna().drop(columns=['year', 'version', 'team', 'continent'])
world_cup_data = world_cup_data.rename(columns={'is_host':'away_host', 'goals_scored_last_4y':'away_scored_4',
       'goals_received_last_4y':'away_received_4', 'wins_last_4y':'away_wins_4', 'losses_last_4y':'away_loss_4',
       'draws_last_4y':'away_draws_4', 'squad_total_market_value_eur':'away_market_value(eur)',
       'fifa_rank_pre_tournament':'away_rank_pretournament', 'squad_avg_age':'away_avg_age'})

print(world_cup_data)