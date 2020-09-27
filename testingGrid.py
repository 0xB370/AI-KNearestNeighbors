import numpy as np
import matplotlib.pyplot as plt
from random import randrange
from matplotlib.colors import ListedColormap
"""
 Pendiente1: tomar como entrada:
{"points" : testPoints, "tags" : tagsPredicted}
testPoint = [[2,4],[3,4] .....]
tagsPredicted = [0,3,4,3....]
para imprimir la salida
"""
xx = np.arange(start = 0, stop = 3, step = 1) #[0,1,2]
yy = np.arange(start = 0, stop = 3, step = 1) #[0,1,2]
X, Y = np.meshgrid(xx,yy)

y = [0.56422, 1.77284, 1.52623]
z = [0.15, 0.3, 0.45]
n = [0, 0, 1, 1, 0]

b = []
Z = [[0, 1, 1], [1, 1, 1], [0, 0, 1]] # predicción
# correlación con la grilla 00 01 02 10 11 12 20 21 22
# print(Z)
# print(xx)
# print(yy)

plt.contourf(X, Y, Z, cmap = ListedColormap(('red', 'green'))) 

# Aplicando un arreglo de colores con itertools (para probar, descomentar la importación)
# colors = itertools.cycle(["r", "b", "g"])
# plt.subplot().scatter(z, y, color=next(colors))
plt.subplot().scatter(z, y, color='red')

plt.show()

"""
Pendiente2 : Imprimir puntos originales
scatter
https://stackoverflow.com/questions/14432557/matplotlib-scatter-plot-with-different-text-at-each-data-point

"""