#!/usr/bin/env python
# coding: utf-8

# In[2]:


pip install xgboost


# In[6]:


import pandas as pd
import xgboost as xgb
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
#data = pd.read_csv(r'C:\Users\Prachi Priya\Downloads\DU _Current\gart- data (1).csv', encoding= 'unicode_escape')
data = pd.read_csv(r'C:\Users\Prachi Priya\Downloads\DU _Current\Amph - data (1).csv', encoding= 'unicode_escape')

# Check for missing values and impute them using KNN
imputer = KNNImputer(n_neighbors=5)
data = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)

# Split the dataset into training, validation, and test sets
train_val, test = train_test_split(data, test_size=0.15, random_state=42)
train, val = train_test_split(train_val, test_size=0.15, random_state=42)

# Standardize the training and validation sets
scaler = StandardScaler()
train = pd.DataFrame(scaler.fit_transform(train), columns=train.columns)
val = pd.DataFrame(scaler.transform(val), columns=val.columns)

# Train the model using XGBoost Regression
model = xgb.XGBRegressor(random_state=42)

# Perform hyperparameter tuning using GridSearchCV or RandomizedSearchCV
params = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 15],
    'learning_rate': [0.01, 0.1, 1],
    'subsample': [0.5, 0.75, 1],
    'colsample_bytree': [0.5, 0.75, 1],
}
grid_search = GridSearchCV(model, param_grid=params, cv=5, n_jobs=-1)
grid_search.fit(val.drop('P(Gpa)', axis=1), val['P(Gpa)'])

# Fit the model on the training set using optimized hyperparameters
model = xgb.XGBRegressor(**grid_search.best_params_, random_state=42)
model.fit(train.drop('P(Gpa)', axis=1), train['P(Gpa)'])

# Evaluate the model's performance using the test set
test = pd.DataFrame(scaler.transform(test), columns=test.columns)
y_pred = model.predict(test.drop('P(Gpa)', axis=1))
rmse = mean_squared_error(test['P(Gpa)'], y_pred, squared=False)
nrmse = rmse / test['P(Gpa)'].mean()
r2 = r2_score(test['P(Gpa)'], y_pred)

# Print the evaluation metrics
print('RMSE:', rmse)
print('NRMSE:', nrmse)
print('R2 score:', r2)


# In[ ]:




