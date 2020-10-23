import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import plotter as plt
from plotter import UtilsFunctions

def validacionCruzada(name_file):
    nombre_archivo = name_file.split('/')
    df = pd.read_csv(nombre_archivo[len(nombre_archivo)-1])
    print(nombre_archivo[len(nombre_archivo)-1])
    if(nombre_archivo[len(nombre_archivo)-1]=='datasets_1846_3197_Social_Network_Ads-3.csv' or nombre_archivo[len(nombre_archivo)-1]=='datasets_1846_3197_Social_Network_Ads-2.csv'):
        x=df.iloc[:,[2,3]].values
        y=df.iloc[:,4].values
        # #Extract all row and columns 3 and 5
        # X = dataset.iloc[:, [2, 3]].values
        # #Extract "Purchased" values (1 if purchased, 0 if not)
        # Y = dataset.iloc[:, 4].values
    else:
        x=df.iloc[:,[0,1]].values
        y=df.iloc[:,2].values
    maxAvg = -1
    resp=[]
    k=range(1,11)
    for i in k:
        mayor=False
        knn = KNeighborsClassifier(n_neighbors=i)
        x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=0,test_size=0.3)
        knn.fit(x_train,y_train)
        kf = KFold(n_splits=5, shuffle=True)
        scores = cross_val_score(knn, x, y, scoring='accuracy', cv=kf, n_jobs=-1)
        # print(scores)
        resp.append([i,np.average(scores),False])
    for value in resp:
        if(maxAvg<0 or maxAvg<=value[1]):
            maxAvg=value[1]
            value[2]=True
    for value in resp:
        if(value[1]!=maxAvg):
            value[2]=False
    print(resp)
    return resp






def crossValidation(name_file, folds=5, kRange=11):
    ##################################################################################################
    # VER PARA HACER QUE TOME EL ARCHIVO IGUAL QUE LA FUNCION PARA GRAFICAR
    ##################################################################################################
    nombre_archivo = name_file.split('/')
    df = pd.read_csv(nombre_archivo[len(nombre_archivo)-1])
    print(nombre_archivo[len(nombre_archivo)-1])
    if(nombre_archivo[len(nombre_archivo)-1]=='datasets_1846_3197_Social_Network_Ads-3.csv' or nombre_archivo[len(nombre_archivo)-1]=='datasets_1846_3197_Social_Network_Ads-2.csv'):
        x=df.iloc[:,[2,3]].values
        y=df.iloc[:,4].values
    else:
        x=df.iloc[:,[0,1]].values
        y=df.iloc[:,2].values
    ##################################################################################################
    ##################################################################################################
    ##################################################################################################
    
    # Definimos el rango de K a calcular
    k=range(1,kRange)
    # Creo un array con los distintos tags (Si en el dataset se tienen clasificaciones con tags 'C1' y 'C2', este array será ['C1', 'C2'])
    etiquetas = []
    newY = y
    while len(newY) > 0:
        etiquetas.append(newY[0])
        newY = list(filter(lambda y : y != newY[0], newY))
    # La funcion accuracy_score solo funciona cuando los tags son números, no strings, por lo que hay que tratar esta condición
    # Armo el array yAcc = yAccuracy cambiando los strings por números
    # TODO: Agregar condicional para que esto se ejecute solo en caso que las etiquetas tengan strings
    yAcc = y
    for i in range(len(yAcc)):
        for j in range(len(etiquetas)):
            if (yAcc[i] == etiquetas[j]):
                yAcc[i] = j
    # Si el arreglo de etiquetas contiene strings, los paso a números para poder usarlas con la función accuracy_score
    # TODO: Agregar condicional para que esto se ejecute solo en caso que las etiquetas tengan strings
    etiquetasAcc = etiquetas
    for i in range(len(etiquetasAcc)):
        etiquetasAcc[i] = i
    # Dividimos los arrays en n folds (predeterminadamente, 5)
    utils = UtilsFunctions()
    splittedX = utils.array_split(data=x, folds=folds)
    splittedY = utils.array_split(data=yAcc, folds=folds)
    # Inicializo los arreglos para devolver la respuesta
    res = []
    promediosArr = []
    # Para cada valor de K en el rango que definimos, se calcula la función KNN
    for kValue in k:
        # Inicializamos el array para las predicciones 
        foldPred = []
        # Iteramos sobre la cantidad de folds, asignando a nuestro x_test e y_test un fold diferente en cada iteración
        for j in range(folds):
            # Armamos los sets de entrenamiento y testeo correspondiente a esta iteración (recordar que en cada una cambia el fold para asignar al set de testeo)
            x_train = []
            y_train = []
            # Como splittedX y splittedY tienen la misma longitud, podemos usar un solo for para tratar ambos
            for k in range(len(splittedX)):
                if (k != j):
                    for m in range(len(splittedX[k])):
                        x_train.append(np.array(splittedX[k][m]))
                        y_train.append(np.array(splittedY[k][m]))
                else:
                    x_test = []
                    y_test = []
                    for m in range(len(splittedX[k])):
                        x_test.append(np.array(splittedX[k][m]))
                        y_test.append(np.array(splittedY[k][m]))
            # Ya obtenidos los sets de entramiento y testeo, le damos el formato para pasarselo a nuestra función KNN y obtener las predicciones
            C = []
            for ix in range(len(etiquetasAcc)):
                C.append([])
            for ix in range(len(y_train)):
                elto = []
                for item in x_train[ix]:
                    elto.append(item)
                for yx in range(len(etiquetasAcc)):
                    if (y_train[ix] == etiquetasAcc[yx]):
                        C[yx].append(elto)
            for ix in range(len(C)):
                C[ix] = np.array(C[ix])
            # Le pasamos el set de entrenamiento ya formateado a nuestra función KNN. El valor de K variará en cada iteración
            knn = plt.knnHelper(*C, k=kValue)
            # Le pasamos el set de testeo a nuestra función KNN
            knn.setXTest(x_test=x_test)
            # Predecimos con nuestra función los tags correspondientes al set de testeo pasado previamente
            tagsPredicted = knn.analyse()
            # Calculamos el puntaje de acierto de los tags predichos por nuestra función y lo anexamos a un array que contendrá todos los puntajes de acierto para cada iteración (recordar que en cada una varía el fold que se le asigna al set de testeo)
            score = accuracy_score(y_test, tagsPredicted)
            foldPred.append(score)
        # Se calcula el promedio de todos los puntajes de acierto obtenidos para un valor de K
        promedio = (sum(foldPred)) / (len(foldPred))
        # Este array de promedios nos simplifica la devolución del K óptimo (el marcar con 1 el que mayor promedio tenga)
        promediosArr.append(promedio)
        # Anexamos a la variable de respuesta el valor de K y el promedio
        res.append([kValue, promedio, False])
    # Obtenemos las posiciones de los promedios más altos (Es decir, los K óptimos - Se puede dar el caso en que haya un empate de promedios más altos, en el cual tendremos más de un K óptimo - )
    posiciones = utils.posicionesValor(arr=promediosArr, valor=max(promediosArr))
    # Se ponen a True los K óptimos
    for index_max in range(len(posiciones)):
        res[index_max][2] = True
    return res