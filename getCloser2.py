import math  

def crearFreqs(tagsOrdered):
    freqs = []
    for i in range(len(tagsOrdered)):
        if(len(freqs)>0):
            agregar = True
            for j in range (len(freqs)):
                if(freqs[j][0] == tagsOrdered[i]):
                    agregar = False
                    break
            if(agregar == True):
                freqs.append([tagsOrdered[i],0])
        else:
            freqs.append([tagsOrdered[i],0])
    return freqs
                
def compararFreqs(freqs):
    maxFreq = freqs[0][1]
    maxLabel = freqs[0][0]
    for i in range(len(freqs)):
        if(freqs[i][1]>maxFreq):
            maxFreq = freqs[i][1]
            maxLabel = freqs[i][0]
    return [maxLabel,maxFreq]


def weightedKNN (tagsOrdered,distance):
    freqs = crearFreqs(tagsOrdered)
    for i in range(len(tagsOrdered)):
        for j in range(len(freqs)):
            if(tagsOrdered[i] == freqs[j][0]):
                freqs[j][1] += (1/distance[i])
    maxLabel = compararFreqs(freqs)
    return maxLabel
