#Import Library
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

#Import Data
df = pd.read_csv("input\R-ICERSC.csv")

#Data Preprocess
#One Hot Encoding
features = ['Recycle','EQP_ID1','EQP_ID2','EQP_ID3','RECIPE_ID3','EQP_ID4','MAINPD_ID_1','MAINPD_ID_2']
train_X = df[features]

from sklearn.preprocessing import LabelEncoder
LE1 = LabelEncoder()
LE2 = LabelEncoder()
LE3 = LabelEncoder()
LE4 = LabelEncoder()
LE5 = LabelEncoder()
LE6 = LabelEncoder()
LE7 = LabelEncoder()

train_X['EQP_ID1'] = LE2.fit_transform(train_X['EQP_ID1'])
train_X['EQP_ID2'] = LE3.fit_transform(train_X['EQP_ID2'])
train_X['EQP_ID3'] = LE3.fit_transform(train_X['EQP_ID3'])
train_X['RECIPE_ID3'] = LE3.fit_transform(train_X['RECIPE_ID3'])
train_X['EQP_ID4'] = LE3.fit_transform(train_X['EQP_ID4'])
train_X['MAINPD_ID_1'] = LE4.fit_transform(train_X['MAINPD_ID_1'])
train_X['MAINPD_ID_2'] = LE5.fit_transform(train_X['MAINPD_ID_2'])

#Transfer target data to log value
train_y = np.log1p(df['PAR_Total'].values)

#Model Training
#LGB
import lightgbm as lgb
from Booster import run_lgb 
from sklearn import model_selection
kf = model_selection.KFold(n_splits=5, shuffle=True, random_state=2017) # split 5 sets for cross validation
pred_test_full = 0
for dev_index, val_index in kf.split(train_X):
    dev_X, val_X = train_X.loc[dev_index,:], train_X.loc[val_index,:]
    dev_y, val_y = train_y[dev_index], train_y[val_index]
    model, evals_result = run_lgb(dev_X, dev_y, val_X, val_y)

# XGB
import xgboost as xgb
from Booster import run_xgb
model_xgb = run_xgb(dev_X, dev_y, val_X, val_y)
print("XGB Training Completed...")

### Feature Importance ###
fig, ax = plt.subplots(figsize=(10,8))
xgb.plot_importance(model_xgb, max_num_features=50, height=0.8, ax=ax)
ax.grid(False)
plt.title("XGBoost - Feature Importance", fontsize=15)
plt.show()

### Feature Importance ###
fig, ax = plt.subplots(figsize=(10,8))
lgb.plot_importance(model, max_num_features=50, height=0.8, ax=ax)
ax.grid(False)
plt.title("LightGBM - Feature Importance", fontsize=15)
plt.show()

