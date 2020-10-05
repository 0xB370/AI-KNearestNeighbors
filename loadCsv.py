import pandas as pd
from sklearn.metrics import accuracy_score
from knnPure import KnnClasifier
from sklearn.model_selection import train_test_split

import numpy as np
import pruebaMeshgrid as msg

dataset = pd.read_csv('datasets_1846_3197_Social_Network_Ads.csv')

""" K = 5
x = df['Age'].to_numpy()
y = df['EstimatedSalary'].to_numpy()
tags = df['Purchased'].to_numpy()
points = []

for i in range(x.size):
    points.append([x[i],y[i]])

trainPoints, testPoints = train_test_split(points, test_size=0.1, shuffle=False)
testTags, tagsExpected = train_test_split(tags, test_size=0.1, shuffle=False)

knn = KnnClasifier()
tagsPredicted = knn.predict(testPoints, trainPoints, testTags, K)

print(tagsExpected)
print(tagsPredicted)

a = accuracy_score(tagsExpected, tagsPredicted)

print(a) """


#Extract all row and columns 3 and 5
X = dataset.iloc[:, [2, 3]].values
#Extract "Purchased" values (1 if purchased, 0 if not)
Y = dataset.iloc[:, 4].values

trainPoints, testPoints = train_test_split(X, test_size=0.1, shuffle=False)
testTags, tagsExpected = train_test_split(Y, test_size=0.1, shuffle=False)

C0 = []
C1 = []

for ix in range(len(testTags)):
    elto = []
    for item in trainPoints[ix]:
        elto.append(item)
    if (Y[ix] == 0):
        C0.append(elto)
    if (Y[ix] == 1):
        C1.append(elto)





# apply kNN with k=1 on the same set of training samples
# Con k=39 ya se comienza a romper y con k=40 ya se va de tema
knn = msg.kAnalysis(np.array(C0), np.array(C1), k=5, distance=0)
knn.prepare_test_samples(low=0, high=100, step=0.5)
knn.analyse()
# Cálculo de precisión
nn = knn.precision()
print(testPoints)
tagsPredicted = nn.predict(np.array(testPoints))
print(tagsPredicted)
print(accuracy_score(tagsExpected, tagsPredicted))
####################
knn.plot()
