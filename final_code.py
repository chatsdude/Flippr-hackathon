def clean_type(row):
    if row=='m.corp.': return 'm.corp'
    elif row=='m.cl.' or row=='m cl': return 'm.cl'
    elif row=='m.b.' or row=='mb': return 'm.b'
    elif row=='n.p.' or row=='np': return 'n.p'
    elif row=='n.p.p.' or row=='npp': return 'n.p.p'
    elif row=='t.p.': return 't.p'
    elif row=='ua': return 'u.a'
    elif row=='t.m.c.': return 't.m.c'
    elif row=='c.t.': return 'c.t'
    elif row=='cmc': return 'c.m.c'
    else: return row
from sklearn import preprocessing
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge

#Reading datasets
df_train = pd.read_excel('train.xlsx')
df_test = pd.read_excel('test.xlsx')


#Dropping some attributes
attributes_to_drop=["City", "Popuation [2001]"]
df_train= df_train.drop(attributes_to_drop,1)
df_test= df_test.drop(attributes_to_drop,1)

#Filling missing values with median
df_train.fillna(df_train.median(), inplace=True)
df_test.fillna(df_test.median(), inplace=True)

#Using Label Encoder for State and SWM in train set

df_train["Type"]=df_train["Type"].apply(clean_type)
df_train["SWM"] = df_train["SWM"].astype('category')
# # Assigning numerical values and storing in SWM column
df_train["SWM"] = df_train["SWM"].cat.codes
df_train = df_train[df_train["SWM"] >=0] 
df_train["Type"] = df_train["Type"].astype('category')
df_train["Type"] = df_train["Type"].cat.codes  
df_train["State"] = df_train["State"].astype('category')
df_train["State"] = df_train["State"].cat.codes  

#Using z-score for removing outliers from train set
from scipy import stats
import numpy as np
z = np.abs(stats.zscore(df_train))
threshold = 3
df_train= df_train[(z < 3).all(axis=1)]

#Using Label Encoder for State and SWM in test set
df_test["Type"]=df_test["Type"].apply(clean_type)
df_test["SWM"] = df_test["SWM"].astype('category')
# # Assigning numerical values and storing in another column
df_test["SWM"] = df_test["SWM"].cat.codes
df_test["Type"] = df_test["Type"].astype('category')
df_test["Type"] = df_test["Type"].cat.codes  
df_test["State"] = df_test["State"].astype('category')
df_test["State"] = df_test["State"].cat.codes  


X_train = df_train.iloc[:,:-1].values
y_train = df_train.iloc[:,-1].values

X_test=df_test

#Using Random ForestRegressor and Random forest GridSearchCV  (MODEL 1)
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor 
regressor = RandomForestRegressor(n_estimators = 100, random_state = 42) 

param_grid = [
    # try 12 (3×4) combinations of hyperparameters
    {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
    # then try 6 (2×3) combinations with bootstrap set as False
    {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
]

# train across 5 folds, that's a total of (12+6)*5=90 rounds of training 
grid_search = GridSearchCV(regressor, param_grid, cv=5,
                            scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)
grid_search_rf_model = grid_search.best_estimator_


#Using GradientboostingRegressor (MODEL 2)
from sklearn.model_selection import KFold
from sklearn import datasets, ensemble
from sklearn.inspection import permutation_importance
params = {'n_estimators': 500,
          'max_depth': 4,
          'min_samples_split': 5,
          'learning_rate': 0.01,
          'loss': 'ls'}
reg = ensemble.GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,max_depth=4, max_features='sqrt', min_samples_leaf=15, min_samples_split=10, loss='huber', random_state=42)

param_grid = dict(n_estimators=np.array([50,100,200,300,400]))
model = GradientBoostingRegressor(random_state=42)
kfold = KFold(n_splits=10, random_state=42)
grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring='neg_mean_squared_error', cv=kfold)
grid_result= grid.fit(X_train, y_train)

means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))

print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

model2 = GradientBoostingRegressor(random_state=42, n_estimators=50)


#Using Kernel Ridge Regression (MODEL 3)
KRR = KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5)


#Using Voting regressor with above 3 models
from sklearn.ensemble import VotingRegressor
randomforest = grid_search_rf_model
GBR= model2
KKR= KRR

voting_reg = VotingRegressor(
    estimators=[('rf', randomforest), ('gbr', GBR),('krr',KRR)])
voting_reg.fit(X_train, y_train)
predictions=voting_reg.predict(X_test)
df = pd.DataFrame(predictions)
df.to_csv('Part1.csv')






#prediction of Foreign visitors is done using simple exponential smoothing time series model

dftestpart2=pd.read_excel('test.xlsx',sheet_name="Foreign_Visitors_TS",skiprows=1)
dftestpart2.fillna(0, inplace=True)
#Using Exponential smoothing formula
dftestpart2["September"] = dftestpart2["August"] + 0.9*(1-0.9)*dftestpart2["July"] + (0.9*(1-0.9)**2)*dftestpart2["June"] + (0.9*(1-0.9)**3)*dftestpart2["May"] + (0.9*(1-0.9)**4)*dftestpart2["April"] 
octfv=dftestpart2["September"]

#Dropping september 1 foreign visitors and inserting october 1 foreign visitors column
attributestodrop=["Foreign Visitors"]
df_test= df_test.drop(attributestodrop,1)
df_test["Foreign Visitors"]=octfv
part2_predictions=voting_reg.predict(df_test)
df = pd.DataFrame(part2_predictions)
df.to_csv('Part2.csv')


