import pandas as pd
from sklearn.metrics import accuracy_score
from knnPure import KnnClasifier
from sklearn.model_selection import train_test_split

import numpy as np
import pruebaMeshgrid as msg

dataset = pd.read_csv('datasets_1846_3197_Social_Network_Ads-3.csv')


class CSVUtilities: 
    def getTupleToPrint(self, dataset):
        

    def getMin(self, dataset):
        X = dataset.iloc[:, [0, 1]].values
        return X.min()

    def getMax(self, dataset):
        X = dataset.iloc[:, [0, 1]].values
        return X.max()

    def getTags(self, dataset):
        Y = dataset.iloc[:, 2].values    
        etiquetas = []
        newY = Y
        while len(newY) > 0:
            etiquetas.append(newY[0])
            newY = list(filter(lambda y : y != newY[0], newY))
        return etiquetas
    

    
    
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

"""
#Extract all row and columns 3 and 5
X = dataset.iloc[:, [2, 3]].values
#Extract "Purchased" values (1 if purchased, 0 if not)
Y = dataset.iloc[:, 4].values


# Creo un array con los distintos tags
etiquetas = []
newY = Y
while len(newY) > 0:
  etiquetas.append(newY[0])
  newY = list(filter(lambda y : y != newY[0], newY))
print(etiquetas)


trainPoints, testPoints = train_test_split(X, test_size=0.1, shuffle=False)
testTags, tagsExpected = train_test_split(Y, test_size=0.1, shuffle=False)


# Relleno y les doy el formato a los arrays para pasarle a la función del knn
C = []
for ix in range(len(etiquetas)):
    C.append([])

for ix in range(len(testTags)):
    elto = []
    for item in trainPoints[ix]:
        elto.append(item)
    for yx in range(len(etiquetas)):
        if (testTags[ix] == etiquetas[yx]):
            C[yx].append(elto)

for ix in range(len(C)):
    C[ix] = np.array(C[ix])




# apply kNN with k=1 on the same set of training samples
# Con k=39 ya se comienza a romper y con k=40 ya se va de tema
knn = msg.kAnalysis(*C, k=5, distance=0)
knn.prepare_test_samples(low=X.min(), high=X.max(), step=0.5)
knn.analyse()
# Cálculo de precisión
nn = knn.precision()
print(testPoints)
tagsPredicted = nn.predict(np.array(testPoints))
print(tagsPredicted)
# ESTA LINEA NO FUNCIONA SI SE USAN TAGS CON STRINGS
# print(accuracy_score(tagsExpected, tagsPredicted))
####################
knn.plot(etiquetas=etiquetas)
"""