from pandas import read_csv, DataFrame
from numpy import ones
import seaborn as sns
import matplotlib.pyplot as plt
from pandasql import sqldf
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.linear_model import HuberRegressor, Lars, LassoLars
from sklearn.linear_model import PassiveAggressiveRegressor, RANSACRegressor
from sklearn.linear_model import SGDRegressor, TheilSenRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import ExtraTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import GradientBoostingRegressor
q = lambda q: sqldf(q, globals())

# Prepare Data
df = read_csv("/Users/peterjmyers/Work/Spot-Check-Regression-Algorithms/housingdata.csv", header=None, names = ['CRIM','INDUS','CHAS','NOX','RM','AGE','RAD','TAX','PTRATIO','B','LSTAT','MEDV'])
X = df.drop(["MEDV"], axis=1).values
scaler = StandardScaler()
X = scaler.fit_transform(X)
scaler2 = MinMaxScaler()
X = scaler2.fit_transform(X)
y = df["MEDV"].values

# Spot Check Algorithms
y_pred = ones(len(y)) * y.mean()
baseline_mse = mean_squared_error(y, y_pred)
print("The baseline mse is {}".format(baseline_mse))

# Improve Results
models = []
# Linear
models.append(("LR",LinearRegression()))
alpha = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
for a in alpha:
  models.append(("Lasso-"+str(a),Lasso(alpha=a)))
  models.append(("Ridge-"+str(a),Ridge(alpha=a)))
for a1 in alpha:
  for a2 in alpha:
	  models.append(("EN-" + str(a1) + '-' + str(a2),ElasticNet(a1, a2)))
models.append(("Huber", HuberRegressor()))
models.append(("Lars", Lars()))
models.append(("LassoLars", LassoLars()))
models.append(("PA", PassiveAggressiveRegressor(max_iter=1000, tol=1e-3)))
models.append(("SGD", SGDRegressor(max_iter=1000, tol=1e-3)))
# Nonlinear
for k in range(1, 21):
    models.append(("KNN-"+str(k), KNeighborsRegressor(n_neighbors=k)))
models.append(("DT", DecisionTreeRegressor()))
models.append(("ExtraTree", ExtraTreeRegressor()))
models.append(("SVMLin", SVR(kernel='linear')))
models.append(("SVMPoly", SVR(kernel='poly')))
models.append(("SVRRbf", SVR(kernel='rbf')))
models.append(("SVRSig", SVR(kernel='sigmoid')))
for c in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
    models.append(("SVR-"+str(c), SVR(C=c)))
bag_models = []
# Bagging
# for name, model in models:
#     bag_models.append(("Bagging" + name, BaggingRegressor(model,max_samples=0.5, max_features=0.5)))
models = models + bag_models
# Linear
models.append(("Ranscac", RANSACRegressor()))
models.append(("Theil", TheilSenRegressor()))
# Ensembles
n_trees = 100
models.append(('AdaBoost', AdaBoostRegressor(n_estimators=n_trees)))
models.append(("Bag", BaggingRegressor(n_estimators=n_trees)))
models.append(("RF", RandomForestRegressor(n_estimators=n_trees)))
models.append(("ExtraTrees", ExtraTreesRegressor(n_estimators=n_trees)))
models.append(("GBM", GradientBoostingRegressor(n_estimators=n_trees)))

names, mses = [], []
results = []
for name, model in models:
    print("name: " + name)
    # try:
    #     with catch_warnings():
    #         filterwarnings("ignore")
    cv_mse = cross_val_score(model, X, y, cv = KFold(n_splits=5, random_state=22), scoring='neg_mean_squared_error')
    names.append(name), mses.append(-1*cv_mse.mean())
    results.append((name,-1*cv_mse.mean()))
    # except:
    #     pass

results.sort(key=lambda tup: tup[1])

for name, error in results[:5]:
    print(name, error)



# models_df = DataFrame({'name': names, 'mse': mses}).sort_values(by=['mse']).iloc[0:]

# ax = sns.barplot(x="name", y="mse", data=models_df)
# ax.set_xticklabels(models_df['name'], rotation=75, fontdict={'fontsize': 12})
# plt.savefig('models.png')
# plt.show()
