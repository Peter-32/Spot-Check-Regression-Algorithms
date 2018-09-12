import pandas as pd
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import missingno as msno
from sklearn.preprocessing import StandardScaler
from pandasql import sqldf
from sklearn.metrics import mean_squared_error
from sklearn import linear_model, neighbors, tree, svm, ensemble
from xgboost import XGBRegressor
from sklearn.pipeline import make_pipeline
from tpot.builtins import StackingEstimator
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import GradientBoostingRegressor


q = lambda q: sqldf(q, globals())
mpl.rcParams['figure.figsize'] = (12.0, 5.0)

# Prepare Data
df = pd.read_csv("housingdata.csv", header=None, names = ['CRIM','INDUS','CHAS','NOX','RM','AGE','RAD','TAX','PTRATIO','B','LSTAT','MEDV'])
X = df.drop(["MEDV"], axis=1).values
scaler = StandardScaler()
X = scaler.fit_transform(X)
y = df["MEDV"].values

# Spot Check Algorithms
y_pred = np.ones(len(y)) * y.mean()
baseline_mse = mean_squared_error(y, y_pred)
print("The baseline mse is {}".format(baseline_mse))

# Improve Results

models = []
# Default Models
models.append(("LR",linear_model.LinearRegression()))
models.append(("SGDRegressor",linear_model.SGDRegressor()))
models.append(("ElasticNet",linear_model.ElasticNet()))
models.append(("ElasticNetLasso",linear_model.ElasticNet(alpha=1.0,l1_ratio=0)))
models.append(("Ridge",linear_model.Ridge()))
models.append(("KNN",neighbors.KNeighborsRegressor()))
models.append(("DT",tree.DecisionTreeRegressor()))
models.append(("SVRLinear",svm.SVR(kernel='linear')))
models.append(("SVRPoly",svm.SVR(kernel='poly')))
models.append(("SVRRbf",svm.SVR(kernel='rbf')))
models.append(("SVRSigmoid",svm.SVR(kernel='sigmoid')))
bag_models = []
# Default Bagging
for name, model in models:
    bag_models.append(("Bagging" + name,ensemble.BaggingRegressor(model,max_samples=0.5, max_features=0.5)))
models = models + bag_models
# Default Ensemble
models.append(("ExtraTreesRegressor",ensemble.ExtraTreesRegressor(n_estimators=50, max_depth=None, min_samples_split=2, random_state=2)))
models.append(("AdaBoost",ensemble.AdaBoostRegressor(n_estimators=50, random_state=2)))
models.append(("GradientBoostingRegressor",ensemble.GradientBoostingRegressor(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)))
models.append(("RandomForestRegressor",ensemble.RandomForestRegressor(n_estimators = 50, max_features="log2", min_samples_leaf=5, criterion="mse",
                                    bootstrap = True,random_state=2)))
models.append(("XGBRegressor",make_pipeline(
StackingEstimator(estimator=linear_model.RidgeCV()),
XGBRegressor(learning_rate=0.1, max_depth=10, min_child_weight=13, n_estimators=40, nthread=1, subsample=0.55)
)))
results = []
names = []
for name, model in models:
    print(name)
    kfold = KFold(n_splits=5, random_state=22)
    cv_result = cross_val_score(model, X, y, cv = kfold, scoring='neg_mean_squared_error')
    names.append(name)
    results.append(cv_result)

fig, ax = plt.subplots(figsize=(16, 16))
ax = sns.boxplot(data=results)
ax.set_xticklabels(names, rotation=80)
plt.plot(np.linspace(-20,120,1000), [-1*baseline_mse]*1000, 'r')
plt.show()
