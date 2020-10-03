import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from knnPure import KnnClasifier
import matplotlib.pyplot as plt
from pruebaMeshgrid import kAnalysis
from sklearn.model_selection import train_test_split
import pruebaMeshgrid as generate_random_points
df = pd.read_csv('datasets-short.csv')

K = 5
x = df['Age'].to_numpy()
y = df['EstimatedSalary'].to_numpy()
tags = df['Purchased'].to_numpy()
points = []
pointsToPlot = {}

for i in range(x.size):
    points.append([float(x[i]), float(y[i])])
       
    if tags[i] in pointsToPlot:
        pointsToPlot[tags[i]].append([float(x[i]), float(y[i])])
    else: 
        pointsToPlot[tags[i]] = [[float(x[i]), float(y[i])]]

result = []
for item in pointsToPlot.items():
    i =  np.array(item[1]) # no funciona
    k = (70 - 10) * np.random.random_sample((11, 2)) + 10 # funciona
    result.append(k)
    
t = tuple(result)


knn = kAnalysis(*t, k=5, distance=0)
knn.prepare_test_samples(low=-1, high=70, step=0.3)
knn.analyse()
knn.plot()
plt.show()
    
       
    

#trainPoints, testPoints = train_test_split(points, test_size=0.1, shuffle=False)
#testTags, tagsExpected = train_test_split(tags, test_size=0.1, shuffle=False)

#knn = KnnClasifier()
#tagsPredicted = knn.predict(testPoints, trainPoints, testTags, K)

#print(tagsExpected)
#print(tagsPredicted)

#a = accuracy_score(tagsExpected, tagsPredicted)

#print(a)