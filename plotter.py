# numpy and matplotlib will be used a lot during the lecture
# if you are familiar with these libraries you may skip this part
# if not - extended comments were added to make it easier to understand

# it is kind of standard to import numpy as np and pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from knnPure import KnnClasifier

# used later to apply different colors in for loops
mpl_colors = ('r', 'b', 'g', 'c', 'm', 'y', 'k', 'w')

# just to overwrite default colab style
plt.style.use('default')
plt.style.use('seaborn-talk')



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


class kNearestNeighbors():
  """k-Nearest Neighbor Classifier"""


  def __init__(self, x, y, k=1):
    """Set distance definition: 0 - L1, 1 - L2"""
    
    self.k = k
    self.x_train = x
    self.y_train = y


  def predict(self, x):
    """Predict and return labels for each feature vector from x

    x -- feature vectors (N x D)
    """
    predictions = []  # placeholder for N labels

    # no. of classes = max label (labels starts from 0)
    # np.amax = Return the maximum of an array or maximum along an axis.
    if isinstance(self.y_train[0], np.ndarray):
      flat_list = []
      for sublist in self.y_train:
          for item in sublist:
              flat_list.append(item)
      self.y_train = flat_list
    nof_classes = np.amax(self.y_train) + 1

    """ knn = KnnClasifier()
    tagsPredicted = knn.predict(x, self.x_train, self.y_train, 1) """


    # loop over all test samples
    for x_test in x:
      # array of distances between current test and all training samples
      distances = np.sum(np.abs(self.x_train - x_test), axis=1)
     

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
    # return tagsPredicted

class kAnalysis():
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

  
    
   

    # save training samples to plot them later
  

    # merge feature vector arrays for NearestNeighbor
    # BETO
    # concatenate([array([0, 0]), array([1, 1]), array([2, 2]), array([3, 3]), array([4, 4])]) = [0 0 1 1 2 2 3 3 4 4]
    x = np.concatenate(x, axis=0)

    # train classifier (knn this time)
    self.nn = kNearestNeighbors(x, y, k)
   
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
    
    


  def plot(self, t='', etiquetas=[]):
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
      plot.plot(*x.T, mpl_colors[i] + 'o', label=etiquetas[i])
    plt.legend()
    
    """ for i, x in enumerate(self.classified_prueba):
      plot.plot(*x.T, mpl_colors[i] + '^') """

    # plot test samples
    # BETO
    # Yo cambiaría el comentario de arriba por "pintando la grilla"
    for i, x in enumerate(self.classified):
      plot.plot(*x.T, mpl_colors[i] + ',')
    plt.show()
  
  def precision(self):
    return self.nn
  
class Plotter: 
    def plotKnnGraphic(self, *tupleToPrint, K, minValue, maxValue, step, etiquetas):
      knn = kAnalysis(*tupleToPrint, k=K, distance=0)
      knn.prepare_test_samples(low=minValue, high=maxValue, step=step)
      knn.analyse()
      knn.plot(etiquetas=etiquetas)




      




"""
l1 = Analysis(X1, X2, X3, distance=0)
# l1.prepare_test_samples(low=(-1), high=3, step=0.01)
l1.prepare_test_samples(low=-1, high=2, step=0.01)
l1.analyse()
l1.plot()

plt.show()
"""

""" # apply kNN with k=1 on the same set of training samples
# Con k=39 ya se comienza a romper y con k=40 ya se va de tema
knn = kAnalysis(X1, X2, X3, k=5, distance=0)
knn.prepare_test_samples(low=-1, high=3, step=0.02)
knn.analyse()
# knn.analyse_prueba([[0.5, 0.5], [0.5, 1], [0.5, 1.5], [1, 0.5], [1, 1], [1, 1.5], [1.5, 0.5], [1.5, 1], [1.5, 1.5]])
knn.plot()
plt.show() """