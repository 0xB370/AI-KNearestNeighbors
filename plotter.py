import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import collections as col
# Paleta de colores para la gráfica
paleta_colores = ('r', 'b', 'g', 'c', 'm', 'y', 'k', 'w')

# Sobreescritura el estilo por defecto
plt.style.use('default')
plt.style.use('seaborn-talk')



def init_plot(x_range=None, y_range=None, x_label="$x_1$", y_label="$x_2$"):
  """Definición de ejes (rango) y labels
  x_range -- [min x, max x]
  y_range -- [min y, max y]
  x_label -- string
  y_label -- string
  """
  _, ejes = plt.subplots(dpi=70)
  # Definición del estilo y color del grid
  ejes.grid(c='0.70', linestyle=':')
  # Definición del rango de los ejes
  ejes.set_xlim(x_range) 
  ejes.set_ylim(y_range)
  # Definición de labels
  ejes.set_xlabel(x_label)
  ejes.set_ylabel(y_label)
  return ejes



class UtilsFunctions():
  def ordenIndices(self, seq):
    return sorted(range(len(seq)), key=seq.__getitem__)
  
  def posicionesValor(self, arr, valor):
    resultado = []
    for ix in range(len(arr)):
      if (arr[ix] == valor):
        resultado.append(ix)
    return resultado
  
  def array_split(self, data, folds):
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



class KnnClassifier():
  """Clasificador k-Nearest Neighbor"""
  def __init__(self, x, y, k=1):   
    self.k = k
    self.x_train = x
    self.y_train = y
    newY = y
    self.tagsEmp = []
    if isinstance(newY[0], np.ndarray):
      for subArr in newY:
        while len(subArr) > 0:
          self.tagsEmp.append(subArr[0])
          subArr = list(filter(lambda y : y != subArr[0], subArr))
    else:
      while len(newY) > 0:
          self.tagsEmp.append(newY[0])
          newY = list(filter(lambda y : y != newY[0], newY))
    self.tagsEmp.append(self.tagsEmp[len(self.tagsEmp)-1] + 1)

  def predict(self, x):
    """Predicción de clase para cada elemento de x
    x -- (N x D)
    """
    if isinstance(self.y_train[0], np.ndarray):
      flat_list = []
      for sublist in self.y_train:
          for item in sublist:
              flat_list.append(item)
      self.y_train = flat_list
    nof_classes = max(range(len(self.y_train)), key=self.y_train.__getitem__) + 1
    predictions = []
    
    for x_test in x:
      # Array de distancias entre el punto de prueba actual (x_test) y todos los puntos de "entrenamiento"
      distances = np.array([]) 
      aux = abs((x_test - self.x_train))
      distances = [sum(l) for l in aux]
      # Redondeamos a 2 decimales los valores de distances
      for ix in range(len(distances)):
        distances[ix] = float("%.2f" % distances[ix])
      votes = []
      for ix in range(nof_classes):
        votes.append(0)
      # votes = np.zeros(nof_classes, dtype=np.int)
      # Búsqueda de los K vecinos más cercanos y votación
      # argsort devuelve los índices que ordenarían un array
      # Por lo tanto, los índices de los vecinos más cercanos
      # El [:self.k] es un slice del array obtenido en np.argsort
      # Es decir, lo deja en k valores
      utils = UtilsFunctions()
      for neighbor_id in utils.ordenIndices(seq=distances)[:self.k]:
        # Este label corresponde a uno de los vecinos más cercanos
        neighbor_label = self.y_train[neighbor_id]
        # Actualización del arreglo de votos
        votes[neighbor_label] += 1
      # El label predicho es el que tiene la mayor cantidad de votos
      # En caso de haber empate, se tomará la clase del punto de menor distancia
      posicion = utils.posicionesValor(arr=votes, valor=max(votes))
      # Si hay más de una clase con conteo de votos máximo
      """ if (len(posicion) > 1):
        # Posición simepre va a ser un array, debido a que hay más de un elemento con el valor máximo de votos. Este arreglo contiene las posiciones del arreglo votes que empataron con la máxima cantidad de votos
        bandera = False
        # Con argsort obtenemos un arreglo con las posiciones de los elementos que van de menor a mayor en el arreglo de distancias
        orden = utils.ordenIndices(seq=distances)
        # Iteramos sobre el arreglo obtenido anteriormente tratando primero los puntos más cercanos, y alejándonos gradualmente en cada iteración
        for i in range(len(orden)):
          # Por cada elemento de distances nos fijamos si la clase del punto en cuestión pertenece a las clases que empataron, las cuales se encuentran en el arreglo posición
          for j in range(len(posicion)):
            # Si la distancia mínima tratada en esta iteración existe en el arreglo de las clases que obtuvieron mayor cantidad de votos (posición), se lo agrega a predictions y se sale de los loops
            if (self.y_train[orden[i]] == posicion[j]):
              predictions.append(self.y_train[orden[i]])
              bandera = True
              break
          if (bandera):
              break
      else:
        predictions.append(max(range(len(votes)), key=votes.__getitem__)) """
      if (len(posicion) > 1):
        predictions.append(self.tagsEmp[len(self.tagsEmp)-1])
      else:
        predictions.append(max(range(len(votes)), key=votes.__getitem__))
    return predictions



class knnHelper():
  """Aplicación de kNearestNeighbor a los puntos de prueba"""

  def __init__(self, *x, k=1):
    """Inicialización del clasificador
    k -- número de vecinos más cercanos
    """
    # Definición de cantidad de clases
    self.nof_classes = len(x) + 1
    # Definición de training samples
    self.x_train = x
    # Creación de array de labels
    y = [i * np.ones(_x.shape[0], dtype=np.int) for i, _x in enumerate(x)]
    y = np.array(y).ravel()
    # Creación de un flat array para NearestNeighbor
    # concatenate([array([0, 0]), array([1, 1]), array([2, 2]), array([3, 3]), array([4, 4])]) = [0 0 1 1 2 2 3 3 4 4]
    x = np.concatenate(x, axis=0)
    # Inicialización de clasificador
    self.nn = KnnClassifier(x, y, k)
   
  def generateGridPoints(self, min=0, max=2, step=0.01):
    """Generación del grid con los puntos de prueba"""
    # Definición del rango
    self.range = [min, max]
    # Definición del grid de [min, max] x [min, max] puntos
    # Ejemplos de np.mgrid:
    # np.mgrid[0:7:5] => array([0, 5])
    # np.mgrid[0:15:5] => array([ 0,  5, 10])
    # np.mgrid[0:15:2] => array([ 0,  2,  4,  6,  8, 10, 12, 14])
    grilla = np.mgrid[min:max+step:step, min:max+step:step]
    # Conversión a array de puntos bidimensionales
    # np.vstack apila arrays verticalmente
    # np.vstack(([1,2,3],[2,3,4])) = array([[1, 2, 3],
    #                                       [2, 3, 4]])
    self.x_test = np.vstack([grilla[0].ravel(), grilla[1].ravel()]).T
  
  def setXTest(self, x_test):
    self.x_test = x_test

  def analyse(self):
    """Ejecución del clasificador sobre los puntos de prueba y separación de los mismos de acuerdo a las respectivas etiquetas"""
    # Búsqueda de etiquetas por puntos de prueba
    self.y_test = self.nn.predict(self.x_test)
    self.classified = []
    # Iteración sobre los labels disponibles
    for tag in range(self.nof_classes):
      # Si el i-ésimo label == label actual -> add test[i]
      clasificacion_i = np.array([self.x_test[i] \
                          for i, t in enumerate(self.y_test) \
                          if t == tag])
      self.classified.append(clasificacion_i)
    return self.y_test

  def plot(self, t='', K=5, etiquetas=[], x_label="X", y_label="Y"):
    """Visualización de los resultados de la clasificación"""
    plot = init_plot(self.range, self.range, x_label=x_label, y_label=y_label)
    plot.set_title(t)
    plot.grid(False)
    # Gráfica de los puntos de prueba y sus respectivas leyendas
    legends = []
    for i, x in enumerate(self.x_train):
      legendClass, = plt.plot(*x.T, paleta_colores[i] + 'o', label=etiquetas[i])
      legends.append(legendClass)
    unclass = mpatches.Patch(color='w', label="Unclassified")
    legends.append(unclass)
    plt.legend(handles=[*legends], loc='upper right', facecolor="lightgrey")
    # Pintando la grilla
    for i, x in enumerate(self.classified):
        if (i == (self.nof_classes - 1)):
          plot.plot(*x.T, 'w' + '+')
        else:
          plot.plot(*x.T, paleta_colores[i] + ',')
    plt.show()

  def plot2(self, arrArgs):
    for arrArg in arrArgs:
      """Visualización de los resultados de la clasificación"""
      plot = init_plot(self.range, self.range, x_label=arrArg[3], y_label=arrArg[4])
      plot.set_title(arrArg[0])
      plot.grid(False)
      # Gráfica de los puntos de prueba y sus respectivas leyendas
      legends = []
      for i, x in enumerate(self.x_train):
        legendClass, = plt.plot(*x.T, paleta_colores[i] + 'o', label=arrArg[2][i])
        legends.append(legendClass)
      unclass = mpatches.Patch(color='w', label="Unclassified")
      legends.append(unclass)
      plt.legend(handles=[*legends], loc='upper right', facecolor="lightgrey")
      # Pintando la grilla
      for i, x in enumerate(self.classified):
        if (i == (self.nof_classes)):
          plot.plot(*x.T, 'k' + ',')
        else:
          plot.plot(*x.T, paleta_colores[i] + ',')
    plt.show()

class Plotter: 
    def plotKnnGraphic(self, *tupleToPrint, K, minValue, maxValue, step, etiquetas, x_label, y_label):
      knn = knnHelper(*tupleToPrint, k=K)
      knn.generateGridPoints(min=minValue, max=maxValue, step=step)
      knn.analyse()
      kStr = str(K)
      knn.plot(t='KNN Classifier with K = '+kStr, K=K, etiquetas=etiquetas, x_label=x_label, y_label=y_label)

    def variasGraficas(self,arrArgs):
      aux = []
      for n in arrArgs:
        auxArr=[]
        maxim = n[len(n)- 1] - 1
        ##aux3 = np.array([n[0],n[1],n[2]])
        for i in range(0,maxim+1):
          auxArr.append(n[i])
        aux3 = np.array(auxArr)
        knn = knnHelper(*aux3, k=n[maxim+1])
        knn.generateGridPoints(min=n[maxim + 2], max=n[maxim + 3], step=n[maxim + 4])
        knn.analyse()
        kStr = str(n[maxim + 1])
        K=n[maxim + 1]
        etiquetas=n[maxim + 5]
        print(etiquetas)
        x_label=n[maxim + 6]
        y_label=n[maxim + 7]
        aux.append(['KNN Classifier with K = '+kStr, K, etiquetas, x_label, y_label, n[len(n)-3],n[len(n)-2]])
      print(aux)
      knn.plot2(aux)

            
