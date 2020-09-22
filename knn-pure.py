#from sklearn.metrics import accuracy_score
import numpy as np
## K Nearest Neighbors 

"""
Pendinte 1 crear una clases KnnClasifier

metodo predict(testPoints,points,tags,K) devuelve {"points" : testPoints, "tags" : tagsPredicted}

metodo distance(p1,p2) devuelva distancia

getClassQuantity(tags) ej: [clase1, clase2, clase1, clase1] return 2 

"""
testPoints = [[31,4]] #testPoints
points = [[31,3],[31,3], [13,3], [54,3]] #points
tags = np.array([4,2,2,4]) # tags
K = 4
tagsPredicted = [] # [3,4,5,6]
def contarClases(tagsOrdered):
    cantidadMaxima = 0
    tag = 0
    tags = []
    arr = []
    for i in range(len(tagsOrdered)):
        cant = 0
        for j in range(len(tagsOrdered)):
            if(tagsOrdered[j] == tagsOrdered[i]):
                cant = cant + 1
        if(cant > cantidadMaxima):
            cantidadMaxima = cant
            tag = tagsOrdered[i]
            tags.append(tag)
            arr = [[cantidadMaxima,tag,i]]
        elif((cant == cantidadMaxima) and (tag != tagsOrdered[i])):
            cantidadMaxima = cant
            tag = tagsOrdered[i]
            if(tag not in tags):
                arr.append([cantidadMaxima,tag,i])
    return arr

def getClosest(distances,tagsOrdered):
    tags = contarClases(tagsOrdered)
    # return tags[0][1]
    distMin = distances[tags[0][2]]
    ganador = [distMin,tags[0][1]]
    if(len(tags)>1):
        for i in range(len(tags)):
            if(distances[tags[i][2]]<distMin):
                distMin = distances[i]
                ganador = [distMin,tags[i][1]]
    return ganador

            
## Iterate through each value in test data 
for val in testPoints:
    euc_dis = []
    ## Finding eucledian distance for all points in training data 
    for point in points:
        euc_dis.append(((val[0]-point[0])**2+(val[1]-point[1])**2)**0.5)
        print(euc_dis)
    temp_target = tags
    ## Use bubble sort to sort the euclidean distances 
    for i in range(len(euc_dis)):
        for j in range(0,len(euc_dis)-i-1):
            if(euc_dis[j+1] < euc_dis[j]):
                euc_dis[j], euc_dis[j+1] = euc_dis[j+1], euc_dis[j]
                ## Sort the classes along with the eucledian distances 
                ## to maintain relevancy 
                temp_target[j], temp_target[j+1] = temp_target[j+1], temp_target[j] 
    
    # slice to get the K first entries
    distances = euc_dis[0:K]
    tagsOrdered = temp_target[0:K]
    print("Desde Aca Arranca")
    print(tagsOrdered)
    print(distances)
    var2 = contarClases(tagsOrdered)
    var1 = getClosest(distances,tagsOrdered)
    print(var2)
    print(var1)

    """
    # PENDIENTE - Implementar un una función getClosest(distances,tagsOrdered) y devuelva  el tag según politica:
        Se elige como clase ganadora a la que tiene major cantidad de elementos
        Si 2 o mas clases coinciden se elige la que tiene el elemnto mas cercano
        
        Ej:
        distances = [2,5,6,7]
        tagsOrdered = [clase1, clase2, clase1, clase1]
        la clase ganadora es la clase1
        
        Ej2:
        distances = [2,5,6,7]
        tagsOrdered = [clase2, clase2, clase1, clase1]
        la clase ganadora es la clase2   
        https://jakevdp.github.io/PythonDataScienceHandbook/03.02-data-indexing-and-selection.html
        
        import pandas as pd
        data = pd.Series([0, 0, 0, 0],
                        index=['class0', 'class1', 'class2', 'class3'])

        data['class0'] = 9 
        data['class0'] = data['class0'] + 1
        data['class0']       
             
    """
    
# prediction = {"points" : testPoints, "tags" : tagsPredicted}
