import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
import plotter as plt
from operator import itemgetter
from plotter import UtilsFunctions

def crossValidation(df):
    
    x=df.iloc[:,[0,1]].values
    y=df.iloc[:,2].values
    folds = int(len(x))
    kRange = int(len(x) - 1)
    
    
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
    acum = 0
    kMax = 0
    for index_max in range(len(posiciones)):
        res[index_max][2] = True
    
    for el in res:
        acum = acum + el[1]
        kMax = el[0]
        
    avg = acum / kMax
    
    print(avg)
    print(kMax)
    print(acum)
    
    resSorted = sorted(res, key=itemgetter(1), reverse=True)
    
    return res