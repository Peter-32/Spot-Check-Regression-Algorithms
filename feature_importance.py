import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import missingno as msno
from pandasql import sqldf
from sklearn.metrics import mean_squared_error
from sklearn import linear_model, neighbors, tree, svm, ensemble
from xgboost import XGBRegressor

from tpot.builtins import StackingEstimator
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler


from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import ExtraTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
pd.set_option('display.width', 5000)
pd.set_option('display.max_columns', 60)
q = lambda q: sqldf(q, globals())
mpl.rcParams['figure.figsize'] = (16.0, 16.0)

pd.options.display.html.table_schema = True
pd.options.display.max_rows = None


# Prepare Data
df = pd.read_csv("/Users/peterjmyers/Work/Spot-Check-Regression-Algorithms/housingdata.csv", header=None, names = ['CRIM','INDUS','CHAS','NOX','RM','AGE','RAD','TAX','PTRATIO','B','LSTAT','MEDV'])
X = df.drop(["MEDV"], axis=1).values
scaler = StandardScaler()
X = scaler.fit_transform(X)
scaler2 = MinMaxScaler()
X = scaler2.fit_transform(X)
y = df["MEDV"].values

# Spot Check Algorithms
y_pred = np.ones(len(y)) * y.mean()
baseline_mse = mean_squared_error(y, y_pred)
print("The baseline mse is {}".format(baseline_mse))
#
# # Improve Results
# models = []
# # Linear
# models.append(("LR",linear_model.LinearRegression()))
# alpha = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
# for a in alpha:
#   models.append("Lasso-"+str(a),Lasso(alpha=a))
#   models.append("Ridge-"+str(a),Ridge(alpha=a))
# for a1 in alpha:
#   for a2 in alpha:
# 	  models.append("EN-" + str(a1) + '-' + str(a2),ElasticNet(a1, a2))
# models.append("Huber", HuberRegressor())
# models.append("Lars", Lars())
# models.append("LassoLars", LassoLars())
# models.append("PA", PassiveAggressiveRegressor(max_iter=1000, tol=1e-3))
# models.append("Ranscac", RANSACRegressor())
# models.append("SGD", SGDRegressor(max_iter=1000, tol=1e-3))
# models.append("Theil", TheilSenRegressor())
# # Nonlinear
# for k in range(1, 21):
#     models.append("KNN-"+str(k), KNeighborsRegressor(n_neighbors=k))
# models.append("Cart", DecisionTreeRegressor())
# models.append("ExtraTree", ExtraTreeRegressor())
# models.append("SVMLin", SVR(kernel='linear'))
# models.append("SVMPoly", SVR(kernel='poly'))
# models.append("SVRRbf", SVR(kernel='rbf'))
# models.append("SVRSig", SVR(kernel='sigmoid'))
# for c in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
#   models['SVR-'+str(c)] = SVR(C=c)
# models.append(("DT",tree.DecisionTreeRegressor()))
# bag_models = []
# # Bagging
# for name, model in models:
#     bag_models.append(("Bagging" + name,ensemble.BaggingRegressor(model,max_samples=0.5, max_features=0.5)))
# models = models + bag_models
# # Ensembles
# n_trees = 100
# models['AdaBoost'] = AdaBoostRegressor(n_estimators=n_trees)
# models['Bag'] = BaggingRegressor(n_estimators=n_trees)
# models['RF'] = RandomForestRegressor(n_estimators=n_trees)
# models['ExtraTrees'] = ExtraTreesRegressor(n_estimators=n_trees)
# models['GBM'] = GradientBoostingRegressor(n_estimators=n_trees)
#
# names, mses = [], []
# for name, model in models:
#     cv_mse = cross_val_score(model, X, y, cv = KFold(n_splits=5, random_state=22), scoring='neg_mean_squared_error')
#     names.append(name), mses.append(-1*cv_mse.mean())
# models_df = pd.DataFrame({'name': names, 'mse': mses}).sort_values(by=['mse']).iloc[0:]
# ax = sns.barplot(x="name", y="mse", data=models_df)
# ax.set_xticklabels(models_df['name'], rotation=75, fontdict={'fontsize': 12})
# plt.savefig('models.png')
# plt.show()

pipe_random_forest = Pipeline([('clf', ensemble.RandomForestRegressor(max_depth=None))])
param_grid = [{'clf__max_features': ['auto', 'sqrt', 'log2', 0.2],
               'clf__n_estimators': [25, 50, 75],
               'clf__min_samples_leaf': [1, 2, 3, 4],
               'clf__criterion': ['mse']}]
gs = GridSearchCV(estimator=pipe_random_forest,
                 param_grid=param_grid,
                 scoring='neg_mean_squared_error',
                 cv=5,
                 n_jobs=-1)
gs = gs.fit(X, y)
print("Best Model: {}".format(gs.best_estimator_))
print("MSE: ".format(-1*np.round(gs.best_score_,1)))
print("Best Features:")
model = gs.best_estimator_.steps[0][1]
X = df.drop(["MEDV"], axis=1)
feats = {}
for feature, importance in zip(X, model.feature_importances_):
    feats[feature] = importance
ginis = pd.DataFrame.from_dict(feats, orient='index').rename(columns={0: 'gini_importance'})
ginis = ginis.query('gini_importance>=0.04')
ax = ginis.sort_values(by='gini_importance').plot(kind='bar', rot=90)
plt.show()
