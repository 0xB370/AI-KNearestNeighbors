import getCloser2 as gc

tagsOrdered = [3,1,5,1,1]
distances = [0.25,1.3,1,0.96,200.21]

result = gc.weightedKNN(tagsOrdered,distances)

print(result)