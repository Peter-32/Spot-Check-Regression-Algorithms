import pandas as pd
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import missingno as msno
from sklearn.preprocessing import StandardScaler
from pandasql import sqldf
q = lambda q: sqldf(q, globals())
mpl.rcParams['figure.figsize'] = (15.0, 5.0)

df = pd.read_csv("housingdata.csv", header=None, names = ['CRIM','INDUS','CHAS','NOX','RM','AGE','RAD','TAX','PTRATIO','B','LSTAT','MEDV'])

print(df.iloc[0:4])
# Check for nulls:
msno.matrix(df)
plt.show()
# Check for infinities:
df_temp = df.fillna(0)
df_temp = df_temp.replace([np.inf, -np.inf], np.nan)
print(df_temp.isnull().values.any())
