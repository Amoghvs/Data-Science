# -*- coding: utf-8 -*-
"""
Created on Wed May 13 00:57:02 2020

@author: Dell
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

from sklearn.datasets import load_breast_cancer


def accuracy_cal(cm):
    tp=cm[0, 0]

    tn=cm[1,1]

    fn=cm[1,0]

    fp=cm[0,1]

    tot=tp+fp+tn+fn

    acc=((tp+tn)/(tot))*100
    return (acc)

cancer = load_breast_cancer()
cancer['data'].shape

x=cancer['data']
y=cancer['target']



X_train, X_test, y_train, y_test = train_test_split(
cancer.data, cancer.target, stratify=cancer.target, random_state=42)
logisticregression = LogisticRegression().fit(X_train, y_train)
print("training set score: %f" % logisticregression.score(X_train, y_train))
print("test set score: %f" % logisticregression.score(X_test, y_test))

Y_pred=logisticregression.predict(X_test)
cm=confusion_matrix(y_test,Y_pred)
cm

accuracy=accuracy_cal(cm)
print( " accuracy of logistic reg is %f" % accuracy)



logisticregression100 = LogisticRegression(C=100).fit(X_train, y_train)
print("training set score: %f" % logisticregression100.score(X_train, y_train))
print("test set score: %f" % logisticregression100.score(X_test, y_test))

logisticregression001 = LogisticRegression(C=0.01).fit(X_train, y_train)
print("training set score: %f" % logisticregression001.score(X_train, y_train))
print("test set score: %f" % logisticregression001.score(X_test, y_test))

plt.plot(logisticregression.coef_.T, 'o', label="C=1")
plt.plot(logisticregression100.coef_.T, 'o', label="C=100")
plt.plot(logisticregression001.coef_.T, 'o', label="C=0.001")
plt.xticks(range(cancer.data.shape[1]), cancer.feature_names, rotation=90)
plt.ylim(-5, 5)
plt.legend()



from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
knn.fit(X_train, y_train)

Y_pred = knn.predict(X_test)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, Y_pred)
cm

accuracy=accuracy_cal(cm)
print( " accuracy of knn is %f" % accuracy)

