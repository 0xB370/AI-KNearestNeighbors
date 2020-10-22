import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
import plotter as plt

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
    print('SE VA A IMPRIMIR X')
    print(x)
    print('SE VA A IMPRIMIR Y')
    print(y)
    k=range(1,11)
    for i in k:
        mayor=False
        # knn = KNeighborsClassifier(n_neighbors=i)
        x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=0,test_size=0.3)
        # knn.fit(x_train,y_train)
        # kf = KFold(n_splits=5, shuffle=True)
        kf = KFold(n_splits=2)
        train, test = kf.split(x)
        print('SE VA A IMPRIMIR LO DEL KF')
        print(train)
        print('SE VA A IMPRIMIR X_TRAIN')
        print(x_train)
        flat_list = []
        for sublist in x_train:
            """ for item in sublist:
                flat_list.append((item)) """
            flat_list.append(np.array(sublist))
        print('SE VA A FLAT_LIST')
        print(flat_list)
        knn = plt.knnHelper(*flat_list, k=5)
        # knn.generateGridPoints(min=0, max=500, step=3)
        knn.generateGridPoints()
        knnn = knn.analyse()
        # scores = knn.analyse()
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



def array_split(data, folds):
    length = int(len(data)/folds) #length of each fold
    res = []
    """ for i in range(folds):
        res.append([]) """
    largo = 0
    for j in range(folds):
        values = []
        for k in range(length):
            values.append(data[largo+k])
        if (j == folds):
            if ((len(data)-1) < (largo+k)):
                contador = largo + k + 1
                while (contador <= (len(data)-1)):
                    values.append(data[contador])
                    contador += 1
        res.append(values)
        largo += length
    return res


def crossValidation(name_file):
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
    folds = 5
    splittedX = array_split(x, folds)
    print('SE VA A IMPRIMIR SPLITTED X')
    print(splittedX[0])
    """ for ix in range(len(splittedX)):
        print('SE VA A IMPRIMIR SPLITTED X')
        print(splittedX[ix]) """
    resp=[]
    k=range(1,11)
    # Creo un array con los distintos tags
    etiquetas = []
    newY = y
    while len(newY) > 0:
        etiquetas.append(newY[0])
        newY = list(filter(lambda y : y != newY[0], newY))
    print('SE VAN A IMPRIMIR LAS ETIQUETAS')
    print(etiquetas)
    # Armo el array yAcc = yAccuracy cambiando los strings por números
    # TODO: Agregar condicional para que esto se ejecute solo en caso que las etiquetas tengan strings
    yAcc = y
    for i in range(len(yAcc)):
        for j in range(len(etiquetas)):
            if (yAcc[i] == etiquetas[j]):
                yAcc[i] = j
    print('SE VAN A IMPRIMIR LAS ETIQUETAS YACC')
    print(yAcc)
    
    etiquetasAcc = etiquetas
    for i in range(len(etiquetasAcc)):
        etiquetasAcc[i] = i
    print('SE VAN A IMPRIMIR LAS ETIQUETASACC')
    print(etiquetasAcc)


    splittedY = array_split(yAcc, folds)

    """ x_train = []
    for k in range(len(splittedX)):
        x_train.append(splittedX[k])
    print(x_train) """
    
    for i in k:
        for j in range(folds):
            x_train = []
            y_train = []
            # Como splittedX y splittedY tienen la misma longitud, podemos usar un solo for para tratar ambos
            for k in range(len(splittedX)):
                if (k != j):
                    for m in range(len(splittedX[k])):
                        print(splittedX[k][m])
                        x_train.append(np.array(splittedX[k][m]))
                        y_train.append(np.array(splittedY[k][m]))
                else:
                    x_test = []
                    y_test = []
                    for m in range(len(splittedX[k])):
                        x_test.append(np.array(splittedX[k][m]))
                        y_test.append(np.array(splittedY[k][m]))
            """ print('SE VA A IMPRIMIR X_TRAIN')
            print(len(x_train))
            print(len(y_train))
            print(x_train) """
            
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

        print('SE VA A IMPRIMIR C')
        print(C)
        knn = plt.knnHelper(*C, k=5)
        # knn.generateGridPoints(min=0, max=500, step=3)
        # knn.generateGridPoints()
        knn.setXTest(x_test=x_test)
        predictions = knn.analyse()
        print(predictions)


    print('SE VA A IMPRIMIR X_TRAIN')
    print(x_train)
    print('SE VA A IMPRIMIR X_TEST')
    print(x_test)
    