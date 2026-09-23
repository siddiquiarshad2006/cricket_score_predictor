import pandas as pd
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_absolute_error
import time

print("Loading dataset_level2.pkl...")
df = pickle.load(open('dataset_level2.pkl', 'rb'))

print("Preprocessing data...")
cities = np.where(df['city'].isnull(), df['venue'].str.split().apply(lambda x: x[0]), df['city'])
df['city'] = cities
df.drop(columns=['venue'], inplace=True)

eligible_cities = df['city'].value_counts()[df['city'].value_counts() > 600].index.tolist()
df = df[df['city'].isin(eligible_cities)].copy()

df['current_score'] = df.groupby('match_id')['runs'].cumsum()

df['over'] = df['ball'].apply(lambda x: str(x).split('.')[0])
df['ball_no'] = df['ball'].apply(lambda x: str(x).split('.')[1])

df['balls_bowled'] = (df['over'].astype('int') * 6) + df['ball_no'].astype('int')
df['balls_left'] = 120 - df['balls_bowled']
df['balls_left'] = df['balls_left'].apply(lambda x: 0 if x < 0 else x)

df['player_dismissed'] = df['player_dismissed'].apply(lambda x: 0 if str(x) == '0' else 1).astype('int')
df['player_dismissed'] = df.groupby('match_id')['player_dismissed'].cumsum()
df['wickets_left'] = 10 - df['player_dismissed']

df['crr'] = (df['current_score'] * 6) / df['balls_bowled']

groups = df.groupby('match_id')
match_ids = df['match_id'].unique()
last_five = []
for m_id in match_ids:
    group = groups.get_group(m_id)
    last_five.extend(group['runs'].rolling(window=30).sum().values.tolist())
df['last_five'] = last_five

total_runs = df.groupby('match_id')['runs'].sum().reset_index()
final_df = total_runs.merge(df, on='match_id')
final_df = final_df[['batting_team', 'bowling_team', 'city', 'current_score', 'balls_left', 'wickets_left', 'crr', 'last_five', 'runs_x']]
final_df.dropna(inplace=True)

print(f"Dataset prepared with {len(final_df)} samples.")

X = final_df.drop(columns=['runs_x'])
y = final_df['runs_x']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

trf = ColumnTransformer([
    ('trf', OneHotEncoder(sparse_output=False, drop='first'), ['batting_team', 'bowling_team', 'city'])
], remainder='passthrough')

pipe = Pipeline(steps=[
    ('step1', trf),
    ('step2', StandardScaler()),
    ('step3', XGBRegressor(n_estimators=1000, learning_rate=0.2, max_depth=12, random_state=1, n_jobs=-1))
])

print("Training XGBoost Pipeline...")
start_time = time.time()
pipe.fit(X_train, y_train)
elapsed = time.time() - start_time
print(f"Training completed in {elapsed:.2f} seconds.")

y_pred = pipe.predict(X_test)
print("R2 Score:", r2_score(y_test, y_pred))
print("MAE:", mean_absolute_error(y_test, y_pred))

print("Saving pipe.pkl...")
with open('pipe.pkl', 'wb') as f:
    pickle.dump(pipe, f)

print("Saved pipe.pkl successfully!")
