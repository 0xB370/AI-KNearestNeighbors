import getCloser2 as gc

tagsOrdered = [3,1,5,1,1]
distances = [1.25,1.3,1.6,0.26,0.21]

result = gc.weightedKNN(tagsOrdered,distances)

print(result)