#!/usr/bin/env python
# coding: utf-8

# In[10]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd

players_df = pd.read_csv ("./data/all_seasons.csv")
players_df = players_df.drop(players_df[players_df.draft_round == 'Undrafted'].index)#Remove undrafted
players_df['draft_round'] = pd.to_numeric(players_df['draft_round'])
players_df

# Load the data
X = players_df[["age", "player_height", "player_weight", "pts", "reb", "ast",
                    "oreb_pct", "dreb_pct", "usg_pct", "ts_pct", "ast_pct"]]
y = players_df["net_rating"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale and normalize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Set up the parameter grid to search over
param_grid = {'n_estimators': [100, 200, 500],
                  'max_depth': [None, 10, 20],
                  'max_features': ['auto', 'sqrt']}

# Instantiate a random forest regression model
rf_model = RandomForestRegressor(random_state=42)

# Create a GridSearchCV object to find the best hyperparameters
grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=5)

# Fit the GridSearchCV object to the training data
grid_search.fit(X_train, y_train)


# In[16]:


# Random Forest Model Function
def predict_net_rating(age, player_height, player_weight, pts, reb, ast, oreb_pct, dreb_pct, usg_pct, ts_pct, ast_pct):
    # Load the data
    X = players_df[["age", "player_height", "player_weight", "pts", "reb", "ast",
                    "oreb_pct", "dreb_pct", "usg_pct", "ts_pct", "ast_pct"]]
    y = players_df["net_rating"]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale and normalize the data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Set up the parameter grid to search over
    param_grid = {'n_estimators': [100, 200, 500],
                  'max_depth': [None, 10, 20],
                  'max_features': ['auto', 'sqrt']}

    # Instantiate a random forest regression model
    rf_model = RandomForestRegressor(random_state=42)

    # Create a GridSearchCV object to find the best hyperparameters
    grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=5)

    # Fit the GridSearchCV object to the training data
    grid_search.fit(X_train, y_train)

    # Make a prediction using the trained model
    input_data = [[age, player_height, player_weight, pts, reb, ast, oreb_pct, dreb_pct, usg_pct, ts_pct, ast_pct]]
    input_data_scaled = scaler.transform(input_data)
    net_rating = grid_search.predict(input_data_scaled)[0]

    return net_rating


# In[15]:


import pickle
pickle.dump(grid_search, open('model.pkl','wb'))


# In[7]:


# Make a prediction using the trained model
#input_data = [[age, player_height, player_weight, pts, reb, ast, oreb_pct, dreb_pct, usg_pct, ts_pct, ast_pct]]
#input_data_scaled = scaler.transform(input_data)
#net_rating = grid_search.predict(input_data_scaled)[0]

