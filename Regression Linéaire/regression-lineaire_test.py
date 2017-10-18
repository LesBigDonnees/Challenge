import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import sklearn

#importation des données
df = pd.read_csv('C:/Users/flori_000/Documents/GitHub/Challenge/data/test.csv', sep=";")
df.head()

from sklearn.linear_model import LinearRegression
modeleReg = LinearRegression()

#création de nos variables
#list_var = df.columns.drop(["tH2","date","mois"])
list_var = df.columns.drop(["tH2","date","mois"])
#print(list_var)
y = df.tH2
X = df[list_var]

#regression lineaire
modeleReg.fit(X,y)
print (modeleReg.intercept_)
print (modeleReg.coef_)

#calcul du R²
modeleReg.score(X,y)

RMSE=np.sqrt(((y-modeleReg.predict(X))**2).sum()/len(y))
#print(RMSE)

#représentation de y en fonction des valeurs prédites
plt.plot(y, modeleReg.predict(X),'.')
plt.show()

#représentation de y en fonction des résidus
plt.plot(y, y-modeleReg.predict(X),'.')
plt.show()