#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 12 21:29:48 2020

@author: Beto-23
"""

#Import necessary librearies


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Load dataset
dataset = pd.read_csv('datasets_1846_3197_Social_Network_Ads.csv')

#Extract all row and columns 3 and 5
X = dataset.iloc[:, [2, 3]].values
#Extract "Purchased" values (1 if purchased, 0 if not)
y = dataset.iloc[:, 4].values

from sklearn.model_selection import train_test_split
#75% of the records used as train data, and 25% as test data
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size = 0.25, 
                                                    random_state = 0)

#Standardize scales
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from sklearn.neighbors import KNeighborsClassifier
# metric='minkowski and p=2 are to use euclidean distance'
#Change n_neighbors value ti change "K"
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)

#Training the model
classifier.fit(X_train, y_train)

#Predict using test data
y_pred = classifier.predict(X_test)

#Compare train with test
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)


#Graphic
from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('KNN (Probando)')
plt.xlabel('Edad')
plt.ylabel('Salario')
plt.legend()
plt.show()










