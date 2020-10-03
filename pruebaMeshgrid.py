# numpy and matplotlib will be used a lot during the lecture
# if you are familiar with these libraries you may skip this part
# if not - extended comments were added to make it easier to understand

# it is kind of standard to import numpy as np and pyplot as plt
import numpy as np
import matplotlib.pyplot as plt

# used later to apply different colors in for loops
mpl_colors = ('r', 'b', 'g', 'c', 'm', 'y', 'k', 'w')

# just to overwrite default colab style
plt.style.use('default')
plt.style.use('seaborn-talk')


def generate_random_points(size=10, low=0, high=1):
  """Generate a set of random 2D points

  size -- number of points to generate
  low  -- min value
  high -- max value
  """
  # random_sample([size]) returns random numbers with shape defined by size
  # e.g.
  # >>> np.random.random_sample((2, 3))
  #
  # array([[ 0.44013807,  0.77358569,  0.64338619],
  #        [ 0.54363868,  0.31855232,  0.16791031]])
  #
  return (high - low) * np.random.random_sample((size, 2)) + low


def init_plot(x_range=None, y_range=None, x_label="$x_1$", y_label="$x_2$"):
  """Set axes limits and labels

  x_range -- [min x, max x]
  y_range -- [min y, max y]
  x_label -- string
  y_label -- string
  """

  # subplots returns figure and axes
  # (in general you may want many axes on one figure)
  # we do not need fig here
  # but we will apply changes (including adding points) to axes
  ##############################################################################################
  # https://hackernoon.com/understanding-the-underscore-of-python-309d1a029edc
  # The underscore is also used for ignoring the specific values. If you don’t need the specific values or the values are not used, just assign the values to underscore.
  _, ax = plt.subplots(dpi=70)

  # set grid style and color
  ax.grid(c='0.70', linestyle=':')

  # set axes limits (x_range and y_range is a list with two elements)
  ax.set_xlim(x_range) 
  ax.set_ylim(y_range)

  # set axes labels
  ax.set_xlabel(x_label)
  ax.set_ylabel(y_label)

  # return axes so we can continue modyfing them later
  return ax


def plot_random_points(style=None, color=None):
  """Generate and plot two (separated) sets of random points

  style -- latter group points style (default as first)
  color -- latter group color (default as first)
  """

  # create a plot with x and y ranges from 0 to 2.5
  ax = init_plot([0, 2.5], [0, 2.5])

  # add two different sets of random points
  # first set = 5 points from [0.5, 1.0]x[0.5, 1.0]
  # second set = 5 points from [1.5, 2.0]x[1.5, 2.0]
  # generate_random_points return a numpy array in the format like
  # [[x1, y1], [x2, y2], ..., [xn, yn]]
  # pyplot.plt take separately arrays with X and Y, like
  # plot([x1, x2, x3], [y1, y2, y3])
  # thus, we transpose numpy array to the format
  # [[x1, x2, ..., xn], [y1, y2, ..., yn]]
  # and unpack it with *
  ###################################################################################################
  # https://stackoverflow.com/questions/5741372/syntax-in-python-t
  # The .T accesses the attribute T of the object, which happens to be a NumPy array. The T attribute is the transpose of the array
  ax.plot(*generate_random_points(5, 0.5, 1.0).T, 'ro')
  ax.plot(*generate_random_points(5, 1.5, 2.0).T, style or 'ro')

  return ax


def plot_an_example(style=None, color=None, label="Class"):
  """Plot an example of supervised or unsupervised learning"""
  ax = plot_random_points(style, color)

  # circle areas related to each set of points
  # pyplot.Circle((x, y), r); (x, y) - the center of a circle; r - radius
  # lw - line width
  ax.add_artist(plt.Circle((0.75, 0.75), 0.5, fill=0, color='r', lw=2))
  ax.add_artist(plt.Circle((1.75, 1.75), 0.5, fill=0, color=color or 'r', lw=2))

  # put group labels
  # pyplot.text just put arbitrary text in given coordinates
  ax.text(0.65, 1.4, label + " I", fontdict={'color': 'r'})
  ax.text(1.65, 1.1, label + " II", fontdict={'color': color or 'r'})





"OUR FIRST ML PROBLEM"

######################################################################################################
######################################################################################################
########################################            ##################################################
########################################  ENTRADAS  ##################################################
########################################            ##################################################
######################################################################################################
######################################################################################################

X1 = np.array([[0.15060145, 0.87236189],
       [0.22472043, 0.50416235],
       [0.41595454, 0.02640976],
       [0.95536859, 0.41989985],
       [0.21814531, 0.25401892],
       [0.97011873, 0.24138364],
       [0.8060411 , 0.50421921],
       [0.08356215, 0.86718716],
       [0.78060313, 0.76874876],
       [0.02626516, 0.24964421],
       [0.53063393, 0.98638295],
       [0.84147405, 0.38878643],
       [0.15224119, 0.56446376],
       [0.38910351, 0.49046922],
       [0.23149766, 0.94132446],
       [0.20079752, 0.89792439],
       [0.00154645, 0.95106262],
       [0.06591183, 0.74084623],
       [0.82877249, 0.61152996],
       [0.50801991, 0.02214336]])
X2 = generate_random_points(20, 1, 2)
# BETO
X3 = generate_random_points(21, 2, 3)
X4 = generate_random_points(20, -1, 0)

# new_point = generate_random_points(1, 0, 2)

# plot = init_plot([0, 2], [0, 2])  # [0, 2] x [0, 2]

######################################################################################################
# https://matplotlib.org/tutorials/introductory/pyplot.html
# For every x, y pair of arguments, there is an optional third argument which is the format string that indicates the color and line type of the plot. The letters and symbols of the format string are from MATLAB, and you concatenate a color string with a line style string. The default format string is 'b-', which is a solid blue line. For example, to plot the above with red circles, you would issue
# plot.plot(*X1.T, 'ro', *X2.T, 'bs', *new_point.T, 'g^')

# BETO
# Mostrar gráfico
# plt.show()




class NearestNeighbor():
  """Nearest Neighbor Classifier"""

  def __init__(self, distance=0):
    """Set distance definition: 0 - L1, 1 - L2"""
    if distance == 0:
      self.distance = np.abs     # absolute value
    elif distance == 1:
      self.distance = np.square  # square root
    else:
      raise Exception("Distance not defined.")


  def train(self, x, y):
    """Train the classifier (here simply save training data)

    x -- feature vectors (N x D)
    y -- labels (N x 1)
    """
    # Los labels serían como los "tags" o las clasificaciones de los datos de entrenamientp
    self.x_train = x
    self.y_train = y


  def predict(self, x):
    """Predict and return labels for each feature vector from x

    x -- feature vectors (N x D)
    """
    predictions = []  # placeholder for N labels

    # loop over all test samples
    for x_test in x:
      # array of distances between current test and all training samples
      # BETO
      # Si en el __init__ se le pasa un distance=0, aplica valor absoluto
      # Si se le pasa distance=1, aplica raíz cuadrada
      # Pasandole axis=1 al np.sum(), lo que se hace es sumar los elementos de las tuplas, es decir, si en el parentesis (self.x_train - x_test) el resultado es [[3,1], [1,5]], np.sum() devolverá [4, 6]
      # Otros ejemplos: 
      # (self.x_train - x_test)= [[0,1], [0,5]] --> np.sum() = array([1, 5])
      # (self.x_train - x_test)= [[0,1], [1,5]] --> np.sum() = array([1, 6])
      # (self.x_train - x_test)= [[3,1], [1,5], [2,7]] --> np.sum() = array([4, 6, 9])
      # Para entender bien el tema de la resta entre x_train y x_test, ver el título The magic of numpy del siguiente enlace
      # https://tomaszgolan.github.io/introduction_to_machine_learning/markdown/introduction_to_machine_learning_01_knn/introduction_to_machine_learning_01_knn/
      distances = np.sum(self.distance(self.x_train - x_test), axis=1)

      # get the closest one
      # BETO
      # np.argmin retorna EL ÍNDICE del menor elemento de un array
      # np.argmin([array([0, 0]), array([1, 1]), array([2, 2]), array([3, 3]), array([4, 4])]) = 0
      min_index = np.argmin(distances)

      # add corresponding label
      predictions.append(self.y_train[min_index])

    return predictions








class Analysis():
  """Apply NearestNeighbor to generated (uniformly) test samples."""

  def __init__(self, *x, distance):
    """Generate labels and initilize classifier

    x -- feature vectors arrays
    distance -- 0 for L1, 1 for L2    
    BETO
    L1 hace referencia a que se usará distance=0, o sea valor absoluto como método para el cáclulo de distancia
    L2 hace referencia a que se usará distance=1, o sea raíz cuadrada como método para el cáclulo de distancia
    """
    # get number of classes
    # BETO
    # len(x) = Longitud del array x
    # nof_classes = number of classes
    self.nof_classes = len(x)

    # create lables array
    # np.ones creates an array of given shape filled with 1 of given type
    # we apply consecutive integer numbers as class labels
    # ravel return flatten array
    # BETO 
    # np.ones entonces, dada una matriz NxD, crea una nueva matriz NxD rellena de unos como elementos
    # Para entender el guión bajo antepuesto a la variable x, ver el siguiente enlace:
    # https://hackernoon.com/understanding-the-underscore-of-python-309d1a029edc
    # Donde dice lo siguiente
    # This convention is used for declaring private variables, functions, methods and classes in a module. Anything with this convention are ignored in from module import *. 
    # Para entender el método shape: 
    # https://stackoverrun.com/es/q/2689881
    # The shape attribute for numpy arrays returns the dimensions of the array. If Y has n rows and m columns, then Y.shape is (n,m). So Y.shape[0] is n.
    # Para entender el funcionamiento del enumerate(): 
    # https://www.geeksforgeeks.org/enumerate-in-python/
    # La función ravel() devuelve un array continuo plano, es decir, le saca las comas y los corchetes de los arrays internos, por ejemplo: 
    #  ravel([array([0, 0]), array([1, 1]), array([2, 2]), array([3, 3]), array([4, 4])]) = [0 0 1 1 2 2 3 3 4 4]
    y = [i * np.ones(_x.shape[0], dtype=np.int) for i, _x in enumerate(x)]
    y = np.array(y).ravel()

    # save training samples to plot them later
    self.x_train = x

    # merge feature vector arrays for NearestNeighbor
    # BETO
    # concatenate([array([0, 0]), array([1, 1]), array([2, 2]), array([3, 3]), array([4, 4])]) = [0 0 1 1 2 2 3 3 4 4]
    x = np.concatenate(x, axis=0)

    # train classifier
    self.nn = NearestNeighbor(distance)
    self.nn.train(x, y)


  def prepare_test_samples(self, low=0, high=2, step=0.01):
    """Generate a grid with test points (from low to high with step)"""
    # remember range
    self.range = [low, high]

    # start with grid of points from [low, high] x [low, high]
    # BETO
    # Ejemplos de np.mgrid
    # np.mgrid[0:7:5] => array([0, 5])
    # np.mgrid[0:15:5] => array([ 0,  5, 10])
    # np.mgrid[0:15:2] => array([ 0,  2,  4,  6,  8, 10, 12, 14])
    grid = np.mgrid[low:high+step:step, low:high+step:step]

    # convert to an array of 2D points
    # BETO 
    # np.vstack apila arrays verticalmente
    # np.vstack(([1,2,3],[2,3,4])) = array([[1, 2, 3],
    #                                       [2, 3, 4]])
    self.x_test = np.vstack([grid[0].ravel(), grid[1].ravel()]).T


  def analyse(self):
    """Run classifier on test samples and split them according to labels."""

    # find labels for test samples 
    self.y_test = self.nn.predict(self.x_test)
    # print(self.x_test)

    self.classified = []  # [class I test points, class II test ...]
    
    # loop over available labels
    for label in range(self.nof_classes):
      # if i-th label == current label -> add test[i]
      # BETO
      # https://stackoverflow.com/questions/38125328/what-does-a-backslash-by-itself-mean-in-python
      # A backslash at the end of a line tells Python to extend the current logical line over across to the next physical line.
      class_i = np.array([self.x_test[i] \
                          for i, l in enumerate(self.y_test) \
                          if l == label])
      self.classified.append(class_i)
    
    


  def plot(self, t=''):
    """Visualize the result of classification"""
    plot = init_plot(self.range, self.range)
    plot.set_title(t)
    plot.grid(False)

    """ print('SE VA A IMPRTIMIR CLASSIFIED')
    print(self.classified)
    print('SE VA A IMPRTIMIR EL TIPO DE DATO DE CLASSIFIED')
    print(type(self.classified[0]))
    print('SE VA A IMPRTIMIR CLASSIFIED_PRUEBA')
    print(self.classified_prueba) """

    # plot training samples
    for i, x in enumerate(self.x_train):
      plot.plot(*x.T, mpl_colors[i] + 'o')
    
    """ for i, x in enumerate(self.classified_prueba):
      plot.plot(*x.T, mpl_colors[i] + '^') """

    # plot test samples
    # BETO
    # Yo cambiaría el comentario de arriba por "pintando la grilla"
    for i, x in enumerate(self.classified):
      plot.plot(*x.T, mpl_colors[i] + ',')
  

  """ def analyse_prueba(self, data_prueba):
    self.classified_prueba = []
    self.x_prueba = np.array(data_prueba)
    self.y_prueba = self.nn.predict(self.x_prueba)
    for label in range(self.nof_classes):
      class_j = np.array([self.x_prueba[i] \
                          for i, l in enumerate(self.y_prueba) \
                          if l == label])
      self.classified_prueba.append(class_j)
    # Ni idea por qué me ponía un último elemento en el array del tipo "array([], dtype=float64)"
    # Eso hacía que se rompa todo e imprima cualquier cosa en los labels del eje y
    # PENDIENTEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE
    # Cuando le agrego X4 al kAnalysis también se rompe
    self.classified_prueba.pop() """
    








class kNearestNeighbors(NearestNeighbor):
  """k-Nearest Neighbor Classifier"""


  def __init__(self, k=1, distance=0):
    """Set distance definition: 0 - L1, 1 - L2"""
    super().__init__(distance)
    self.k = k


  def predict(self, x):
    """Predict and return labels for each feature vector from x

    x -- feature vectors (N x D)
    """
    predictions = []  # placeholder for N labels

    # no. of classes = max label (labels starts from 0)
    # np.amax = Return the maximum of an array or maximum along an axis.
    if isinstance(self.y_train[0], np.ndarray):
      print(type(self.y_train[0]))
      flat_list = []
      for sublist in self.y_train:
          for item in sublist:
              flat_list.append(item)
      self.y_train = flat_list
    nof_classes = np.amax(self.y_train) + 1

    # loop over all test samples
    for x_test in x:
      # array of distances between current test and all training samples
      distances = np.sum(self.distance(self.x_train - x_test), axis=1)

      # placeholder for labels votes
      # np.zeros = Return a new array of given shape and type, filled with zeros.
      votes = np.zeros(nof_classes, dtype=np.int)

      # find k closet neighbors and vote
      # argsort returns the indices that would sort an array
      # so indices of nearest neighbors
      # we take self.k first
      # BETO
      # El [:self.k] creo que es un slice del array obtenido en np.argsort
      # Es decir, lo deja en k valores
      # https://stackoverflow.com/questions/39241529/what-is-the-meaning-of-in-python
      for neighbor_id in np.argsort(distances)[:self.k]:
        # this is a label corresponding to one of the closest neighbor
        neighbor_label = self.y_train[neighbor_id]
        # which updates votes array
        votes[neighbor_label] += 1

      # predicted label is the one with most votes
      predictions.append(np.argmax(votes))

    return predictions








class kAnalysis(Analysis):
  """Apply kNearestNeighbor to generated (uniformly) test samples."""

  def __init__(self, *x, k=1, distance=1):
    """Generate labels and initilize classifier

    x -- feature vectors arrays
    k -- number of nearest neighbors
    distance -- 0 for L1, 1 for L2    
    """
    # get number of classes
    self.nof_classes = len(x)

    # create lables array
    y = [i * np.ones(_x.shape[0], dtype=np.int) for i, _x in enumerate(x)]
    y = np.array(y).ravel()

    # save training samples to plot them later
    self.x_train = x

    # merge feature vector arrays for NearestNeighbor
    x = np.concatenate(x, axis=0)

    # train classifier (knn this time)
    self.nn = kNearestNeighbors(k, distance)
    self.nn.train(x, y)








"""
l1 = Analysis(X1, X2, X3, distance=0)
# l1.prepare_test_samples(low=(-1), high=3, step=0.01)
l1.prepare_test_samples(low=-1, high=2, step=0.01)
l1.analyse()
l1.plot()

plt.show()
"""

# apply kNN with k=1 on the same set of training samples
# Con k=39 ya se comienza a romper y con k=40 ya se va de tema
knn = kAnalysis(X1, X2, X3, k=5, distance=0)
knn.prepare_test_samples(low=-1, high=3, step=0.02)
knn.analyse()
# knn.analyse_prueba([[0.5, 0.5], [0.5, 1], [0.5, 1.5], [1, 0.5], [1, 1], [1, 1.5], [1.5, 0.5], [1.5, 1], [1.5, 1.5]])
knn.plot()
plt.show()