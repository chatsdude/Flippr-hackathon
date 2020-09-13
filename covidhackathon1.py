from sklearn import preprocessing
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
#Reading datasets
df_train = pd.read_excel('train.xlsx')
df_test = pd.read_excel('test.xlsx')


#dropping some attributes
attributes_to_drop=["City", "Popuation [2001]","Water Purity"]
df_train= df_train.drop(attributes_to_drop,1)
df_test= df_test.drop(attributes_to_drop,1)

#Filling missing values
df_train.fillna(df_train.median(), inplace=True)
df_test.fillna(df_test.median(), inplace=True)

#taking care of states
#ct=ColumnTransformer(transformers=[('encoder', OneHotEncoder(),[0])],remainder='passthrough')
#df_train=np.array(ct.fit_transform(df_train))
#df_test=np.array(ct.fit_transform(df_test))

df_train=pd.DataFrame(df_train)
print(type(df_train))

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
df_train["Type"]=df_train["Type"].apply(clean_type)
df_train["SWM"] = df_train["SWM"].astype('category')
# # Assigning numerical values and storing in another column
df_train["SWM"] = df_train["SWM"].cat.codes
df_train = df_train[df_train["SWM"] >=0] 
df_train["Type"] = df_train["Type"].astype('category')
df_train["Type"] = df_train["Type"].cat.codes  
df_train["State"] = df_train["State"].astype('category')
df_train["State"] = df_train["State"].cat.codes  
attributes=list(df_train.columns) 
from scipy import stats
import numpy as np
z = np.abs(stats.zscore(df_train))
#print(z)
threshold = 3
df_train= df_train[(z < 3).all(axis=1)]
X= df_train.iloc[:,:-1].values
y= df_train.iloc[:,-1].values
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.fit_transform(X_test)
def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())
from sklearn.ensemble import RandomForestRegressor 
  
# create regressor object 
regressor = RandomForestRegressor(n_estimators = 100, random_state = 42) 
  
# fit the regressor with x and y data 
regressor.fit(X_train,y_train)  
y_pred = regressor.predict(X_test)
#print('Accurcy: {:.4f}'.format(regressor.score(X_test, y_test)))
X_test=pd.DataFrame(X_test)
y_test=pd.DataFrame(y_test)

from sklearn.metrics import mean_squared_error
predictions = regressor.predict(X_train)
rf_mse = mean_squared_error(y_train,predictions)
rf_rmse = np.sqrt(rf_mse)
print("RANDOM FOREST RMSE = " ,rf_rmse)
from sklearn.model_selection import cross_val_score

forest_scores = cross_val_score(regressor, X_train , y_train,
                                scoring="neg_mean_squared_error", cv=10)
forest_rmse_scores = np.sqrt(-forest_scores)
print("CROSS VAL SCORE RANDOM FOREST = ")
display_scores(forest_rmse_scores)

# print('''__________________________________________________________
      
#       ''')
# from sklearn.tree import DecisionTreeRegressor

# tree_reg = DecisionTreeRegressor(random_state=42)
# tree_reg.fit(X_train, y_train)
# predictions = tree_reg.predict(X_train)
# tree_mse = mean_squared_error(y_train, predictions)
# tree_rmse = np.sqrt(tree_mse)
# print("DECISION TREE REGRESSOR RMSE = ",tree_rmse)
# from sklearn.model_selection import cross_val_score

# tree_reg = DecisionTreeRegressor(random_state=42)
# scores = cross_val_score(tree_reg, X_train, y_train,
#                          scoring="neg_mean_squared_error", cv=10)
# tree_rmse_scores = np.sqrt(-scores)

# print("CROSS VAL SCORE DEC TREE = ")
# display_scores(tree_rmse_scores)

# import xgboost as xgb
# xg_reg = xgb.XGBRegressor(objective ='reg:linear', colsample_bytree = 0.3, learning_rate = 0.1,
#                 max_depth = 5, alpha = 10, n_estimators = 10)
# xg_reg.fit(X_train,y_train)
# #predictions = xg_reg.predict(X_test)
# #xg_mse = mean_squared_error(y_test, predictions)
# #xg_rmse = np.sqrt(xg_mse)
# #print("XGBOOST RMSE = ",xg_rmse)
# from sklearn.model_selection import cross_val_score

# #xg_reg = XGBRegressor(random_state=42)
# scores = cross_val_score(xg_reg, X_train, y_train,
#                          scoring="neg_mean_squared_error", cv=10)
# xg_rmse_scores = np.sqrt(-scores)

# #print("CROSS VAL SCORE XGBOOST = ")
# #display_scores(tree_rmse_scores)
from sklearn.model_selection import GridSearchCV

param_grid = [
    # try 12 (3×4) combinations of hyperparameters
    {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
    # then try 6 (2×3) combinations with bootstrap set as False
    {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
]

forest_reg = RandomForestRegressor(random_state=42)
# train across 5 folds, that's a total of (12+6)*5=90 rounds of training 
grid_search = GridSearchCV(regressor, param_grid, cv=5,
                           scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)
print(grid_search.best_params_)
print(grid_search.best_estimator_)

feature_importances = grid_search.best_estimator_.feature_importances_
#extra_attribs = ["rooms_per_hhold", "pop_per_hhold", "bedrooms_per_room"]
#attributes=[]
v=sorted(zip(feature_importances, attributes), reverse=True)
final_model = grid_search.best_estimator_



final_predictions = final_model.predict(X_test)

final_mse = mean_squared_error(y_test, final_predictions)
final_rmse = np.sqrt(final_mse)
print(final_rmse)