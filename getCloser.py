def contarClases(tagsOrdered):
    cantidadMaxima = 0
    tag = 0
    tags = []
    arr = []
    for i in range(len(tagsOrdered)):
        cant = 0
        for j in range(len(tagsOrdered)):
            if(tagsOrdered[j] == tagsOrdered[i]):
                cant = cant + 1
        if(cant > cantidadMaxima):
            cantidadMaxima = cant
            tag = tagsOrdered[i]
            tags.append(tag)
            arr = [[cantidadMaxima,tag,i]]
        elif((cant == cantidadMaxima) and (tag != tagsOrdered[i])):
            cantidadMaxima = cant
            tag = tagsOrdered[i]
            if(tag not in tags):
                arr.append([cantidadMaxima,tag,i])
    return arr

def getClosest(distances,tagsOrdered):
    tags = contarClases(tagsOrdered)
    # return tags[0][1]
    distMin = distances[tags[0][2]]
    ganador = [distMin,tags[0][1]]
    
    if(len(tags)>1):
        for i in range(len(tags)):
            if(distances[tags[i][2]]<distMin):
                distMin = distances[i]
                ganador = [distMin,tags[i][1]]
    
    return ganador


# https://www.geeksforgeeks.org/weighted-k-nn/#:~:text=In%20weighted%20kNN%2C%20the%20nearest,points%20which%20are%20farther%20away.&text=The%20simple%20function%20which%20is%20used%20is%20the%20inverse%20distance%20function.