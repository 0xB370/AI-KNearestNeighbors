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



class kNearestNeighbors():
  """Clasificador k-Nearest Neighbor"""
  def __init__(self, x, y, k=1):   
    self.k = k
    self.x_train = x
    self.y_train = y

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
    nof_classes = np.amax(self.y_train) + 1
    predictions = []
    for x_test in x:
      # Array de distancias entre el punto de prueba actual (x_test) y todos los puntos de "entrenamiento"
      distances = np.sum(np.abs(self.x_train - x_test), axis=1)
      # np.zeros = Devuelve un nuevo array relleno de ceros
      votes = np.zeros(nof_classes, dtype=np.int)
      # Búsqueda de los K vecinos más cercanos y votación
      # argsort devuelve los índices que ordenarían un array
      # Por lo tanto, los índices de los vecinos más cercanos
      # El [:self.k] es un slice del array obtenido en np.argsort
      # Es decir, lo deja en k valores
      for neighbor_id in np.argsort(distances)[:self.k]:
        # Este label corresponde a uno de los vecinos más cercanos
        neighbor_label = self.y_train[neighbor_id]
        # Actualización del arreglo de votos
        votes[neighbor_label] += 1
      # El label predicho es el que tiene la mayor cantidad de votos
      # En caso de haber empate, se tomará la clase del punto de menor distancia
      conteo = col.Counter(votes)
      # Si hay más de una clase con conteo de votos máximo
      if (conteo[np.max(votes)] > 1):
        # Posición simepre va a ser un array, debido a que hay más de un elemento con el valor máximo de votos. Este arreglo contiene las posiciones del arreglo votes que empataron con la máxima cantidad de votos
        posicion = np.array(np.where(votes == np.max(votes))[0])
        bandera = False
        # Con argsort obtenemos un arreglo con las posiciones de los elementos que van de menor a mayor en el arreglo de distancias
        orden = np.argsort(distances)
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
        predictions.append(np.argmax(votes))
      # predictions.append(np.argmax(votes))
    return predictions



class kAnalysis():
  """Aplicación de kNearestNeighbor a los puntos de prueba"""

  def __init__(self, *x, k=1):
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
    # concatenate([array([0, 0]), array([1, 1]), array([2, 2]), array([3, 3]), array([4, 4])]) = [0 0 1 1 2 2 3 3 4 4]
    x = np.concatenate(x, axis=0)
    # Inicialización de clasificador
    self.nn = kNearestNeighbors(x, y, k)
   
  def prepare_test_samples(self, min=0, max=2, step=0.01):
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
    kValue = mpatches.Patch(color='cornflowerblue', label="K="+str(K))
    legends.append(kValue)
    plt.legend(handles=[*legends], loc='upper right')
    # Pintando la grilla
    for i, x in enumerate(self.classified):
      plot.plot(*x.T, paleta_colores[i] + ',')
    plt.show()



class Plotter: 
    def plotKnnGraphic(self, *tupleToPrint, K, minValue, maxValue, step, etiquetas, x_label, y_label):
      knn = kAnalysis(*tupleToPrint, k=K)
      knn.prepare_test_samples(min=minValue, max=maxValue, step=step)
      knn.analyse()
      knn.plot(t='KNN Classifier', K=K, etiquetas=etiquetas, x_label=x_label, y_label=y_label)
