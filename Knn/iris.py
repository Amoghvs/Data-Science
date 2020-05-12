# -*- coding: utf-8 -*-
"""
Created on Tue May 12 13:28:29 2020

@author: Dell
"""


import pandas as pd
from sklearn.datasets import load_iris
iris=load_iris()
import numpy as np
import matplotlib.pyplot as plt

iris.keys()
iris['target_names']
iris['feature_names']

iris['data'].shape

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(iris['data'], iris['target'],
 random_state=0)

fig, ax = plt.subplots(3, 3, figsize=(15, 15))
plt.suptitle("iris_pairplot")
for i in range(3):
    for j in range(3):
        ax[i, j].scatter(X_train[:, j], X_train[:, i + 1], c=y_train, s=60)
        ax[i, j].set_xticks(())
        ax[i, j].set_yticks(())
        if i == 2:
            ax[i, j].set_xlabel(iris['feature_names'][j])
            if j == 0:
                ax[i, j].set_ylabel(iris['feature_names'][i + 1])
                if j > i:
                    ax[i, j].set_visible(False)

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=3)

knn.fit(X_train, y_train)
KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
 metric_params=None, n_jobs=1, n_neighbors=1, p=2,
 weights='uniform')

prediction = knn.predict(X_train)
prediction

y_pred = knn.predict(X_test)
np.mean(y_pred==y_test)

knn.score(X_test,y_test)

