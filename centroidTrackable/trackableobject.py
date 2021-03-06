class TrackableObject:
    def __init__(self, objectID, centroid):
        # almacenamos el objectID, e inicializamos una lista de
        # centroides utilizando el centroide actual
        self.objectID = objectID
        self.centroids = [centroid]

        # Inicilizamos un booleano, para indicar si el objeto
        # ya fue contado o no
        self.counted = False