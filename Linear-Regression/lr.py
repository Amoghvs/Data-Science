# -*- coding: utf-8 -*-
"""
Created on Mon May 11 17:56:46 2020

@author: Dell
"""


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import sys
import sklearn.metrics as metrics
from sklearn.metrics import mean_squared_error
sys.setrecursionlimit(2500)

sys.getrecursionlimit()
data=pd.read_csv("F:\IIT\Github\Data-Science\Linear-Regression\winequality-red.csv")

data.describe()
data.isna().sum()

features = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
       'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
       'pH', 'sulphates', 'alcohol']

x = data[features]
y = data['quality']
#plotting features vs quality

sns.heatmap(data.corr())
plt.show()


x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=3)

regressor = LinearRegression()
regressor.fit(x_train,y_train)

print(regressor.coef_)

train_pred = regressor.predict(x_train)
print(train_pred)
test_pred = regressor.predict(x_test) 
print(test_pred)


# calculating rmse
train_rmse = mean_squared_error(train_pred, y_train) ** 0.5
print(train_rmse)
test_rmse = mean_squared_error(test_pred, y_test) ** 0.5
print(test_rmse)
# rounding off the predicted values for test set
predicted_data = np.round_(test_pred)
print(predicted_data)
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, test_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, test_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, test_pred)))

accuracy = regressor.score(x_test, y_test)
"Accuracy: {}%".format(int(round(accuracy * 100)))