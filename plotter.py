import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import collections as col
# Paleta de colores para la gráfica
paleta_colores = ('r', 'b', 'g', 'c', 'm', 'y', 'k')

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
    length = int(len(data)/folds) # Tamaño de cada fold
    res = []
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
  def __init__(self, x, y, k=1, etiquetas=None):   
    self.k = k
    self.x_train = x
    self.y_train = y
    if (etiquetas is None):
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
    else:
      self.tagsEmp = etiquetas

  def predict(self, x, etiquetas=None):
    """Predicción de clase para cada elemento de x
    x -- (N x D)
    """
    if isinstance(self.y_train[0], np.ndarray):
      self.y_train = np.concatenate(self.y_train, axis=0)
    if (etiquetas is None):
      nof_classes = max(range(len(self.y_train)), key=self.y_train.__getitem__) + 2
    else:
      nof_classes = len(etiquetas) + 1
    predictions = []
    for x_test in x:
      # Array de distancias entre el punto de prueba actual (x_test) y todos los puntos de "entrenamiento"
      distances = np.array([]) 
      aux = abs((x_test - self.x_train))
      distances = [sum(l) for l in aux]
      votes = []
      for ix in range(nof_classes):
        votes.append(0)
      # Búsqueda de los K vecinos más cercanos y votación
      utils = UtilsFunctions()
      for neighbor_id in utils.ordenIndices(seq=distances)[:self.k]:
        # Este label corresponde a uno de los vecinos más cercanos
        neighbor_label = self.y_train[neighbor_id]
        # Actualización del arreglo de votos
        votes[neighbor_label] += 1
      # El label predicho es el que tiene la mayor cantidad de votos
      # En caso de haber empate, se le asignará una clase "Unclassified"
      posicion = utils.posicionesValor(arr=votes, valor=max(votes))
      # Si hay más de una clase con conteo de votos máximo
      if (len(posicion) > 1):
        predictions.append(self.tagsEmp[len(self.tagsEmp)-1])
      else:
        predictions.append(max(range(len(votes)), key=votes.__getitem__))
    return predictions



class knnHelper():
  """Aplicación de kNearestNeighbor a los puntos de prueba"""

  def __init__(self, *x, k=1, etiquetas=None):
    """Inicialización del clasificador
    k -- número de vecinos más cercanos
    """
    # Definición de cantidad de clases
    self.nof_classes = len(x)
    # Definición de training samples
    self.x_train = x
    # Creación de array de labels
    y = [i * np.ones(_x.shape[0], dtype=np.int) for i, _x in enumerate(x)]
    y = np.array(y).ravel()
    # Creación de un flat array para NearestNeighbor
    x = np.concatenate(x, axis=0)
    # Inicialización de clasificador
    if (etiquetas is None):
      self.nn = KnnClassifier(x, y, k)
    else:
      self.nn = KnnClassifier(x, y, k, etiquetas=etiquetas)
    # self.nn = KnnClassifier(x, y, k)
   
  def generateGridPoints(self, min=0, max=2, step=0.01):
    """Generación del grid con los puntos de prueba"""
    # Definición del rango
    self.range = [min, max]
    # Definición del grid de [min, max] x [min, max] puntos
    grilla = np.mgrid[min:max+step:step, min:max+step:step]
    # Conversión a array de puntos bidimensionales
    self.x_test = np.vstack([grilla[0].ravel(), grilla[1].ravel()]).T
  
  def setXTest(self, x_test):
    self.x_test = x_test

  def analyse(self, etiquetas=None):
    """Ejecución del clasificador sobre los puntos de prueba y separación de los mismos de acuerdo a las respectivas etiquetas"""
    # Búsqueda de etiquetas por puntos de prueba
    self.y_test = self.nn.predict(self.x_test, etiquetas=etiquetas)
    self.classified = []
    # Iteración sobre los labels disponibles
    for tag in range(self.nof_classes):
      # Si el i-ésimo label == label actual -> add test[i]
      clasificacion_i = np.array([self.x_test[i] \
                          for i, t in enumerate(self.y_test) \
                          if t == tag])
      self.classified.append(clasificacion_i)
    return self.y_test
  
  def getClassified(self):
    return self.classified

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
      if (i == (self.nof_classes)):
        plot.plot(*x.T, 'w' + '+')
      else:
        plot.plot(*x.T, paleta_colores[i] + 'h', alpha=0.4)
    fig = plt.figure(1)
    fig.canvas.set_window_title('KNN with K=' + str(K))
    plt.show()

  def plot2(self, arrArgs):
    for ix, arrArg in enumerate(arrArgs):
      """Visualización de los resultados de la clasificación"""
      print('Se está procesando el gráfico ' + str(ix+1))
      maxim = arrArg[9]
      n = arrArg[8]
      knn = arrArg[7]
      knn.generateGridPoints(min=n[maxim + 2], max=n[maxim + 3], step=n[maxim + 4])
      range = [n[maxim + 2], n[maxim + 3]]
      knn.analyse()
      classified = knn.getClassified()
      plot = init_plot(range, range, x_label=arrArg[3], y_label=arrArg[4])
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
      for i, x in enumerate(classified):
        if (i == (self.nof_classes)):
          plot.plot(*x.T, 'w' + '+')
        else:
          plot.plot(*x.T, paleta_colores[i] + 'h', alpha=0.4)
      fig = plt.figure(ix+1)
      fig.canvas.set_window_title('KNN with K=' + str(n[maxim+1]))
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
      for i in range(0,maxim+1):
        auxArr.append(n[i])
      aux3 = np.array(auxArr)
      knn = knnHelper(*aux3, k=n[maxim+1])
      kStr = str(n[maxim + 1])
      K=n[maxim + 1]
      etiquetas=n[maxim + 5]
      x_label=n[maxim + 6]
      y_label=n[maxim + 7]
      aux.append(['KNN Classifier with K = '+kStr, K, etiquetas, x_label, y_label, n[len(n)-3],n[len(n)-2], knn, n, maxim])
    knn.plot2(aux)

            
