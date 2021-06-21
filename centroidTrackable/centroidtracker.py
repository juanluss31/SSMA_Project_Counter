# coding=utf-8
from scipy.spatial import distance as dist
from collections import OrderedDict
import numpy as np

class CentroidTracker:
    def __init__(self, maxDisappeared=50, maxDistance=50):
        # Inicializamos el próximo Object ID único, junto a dos diccionarios ordenados,
        # que se utilizarán para mantener constancia de los objetos que se están siguiendo
        # (y calculando su centroide), y para mantener dichos objetos que ya no son visibles
        # durante un periodo de tiempom antes de marcarlos como perdidos.
        self.nextObjectID = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()

        # Cual es el número de frames consecutivos máximo que un objeto puede estar
        # desaparecido antes de ser desregistrado como objeto.
        self.maxDisappeared = maxDisappeared

        self.maxDistance = maxDistance

    def register(self, centroid):
        # Cuando registramos un objeto, utilizamos el siguiente ID disponible
        # para almacenar su centroide.
        self.objects[self.nextObjectID] = centroid
        self.disappeared[self.nextObjectID] = 0
        self.nextObjectID += 1

    def deregister(self, objectID):
        # Una vez se ha dado por perdido un objeto, desregistramos su ID.
        del self.objects[objectID]
        del self.disappeared[objectID]

    def update(self, rects):
        # Comprobamos si la lista de bounding boxes (rects) está vacía.
        if len(rects) == 0:
            # Si teníamos algún objeto marcado como tracked, lo marcamos como desaparecido.
            for objectID in list(self.disappeared.keys()):
                self.disappeared[objectID] += 1
                # Si hemos alcanzado el máximo número de frames consecutivos,
                # desregistramos el objeto
                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)
                # No existen centroides para trackear.
            return self.objects

        # Inicializamos un array para los centroides recibidos en el frame actual, y lo
        # inicializamos a 0
        inputCentroids = np.zeros((len(rects), 2), dtype="int")

        # Iteramos sobre las bounding boxes
        for (i, (startX, startY, endX, endY)) in enumerate(rects):
            # Calculamos el centroide
            cX = int((startX + endX) / 2.0)
            cY = int((startY + endY) / 2.0)
            inputCentroids[i] = (cX, cY)

        # Si no estamos trackeando ningun objeto, registramos los nuevos centroides
        if len(self.objects) == 0:
            for i in range(0, len(inputCentroids)):
                self.register(inputCentroids[i])

        # Si ya estamos trackeando algún objeto, primero intentamos emparejar
        # los nuevos centroides con los que eisten ya de los objetos tackeados.
        else:
            # Obtenemos los IDs y centroides de los objetos
            objectIDs = list(self.objects.keys())
            objectCentroids = list(self.objects.values())

            # Calculamos la distancia entre cada par de centroide trackeado y
            # de nuevos centroides. Nuestro objetivo es emparejar un nuevo centroide con alguno
            # de los existentes.
            D = dist.cdist(np.array(objectCentroids), inputCentroids)

            # Para poder emparejarlos, debemos encontrar el valor más pequeño en cada fila,
            # y entonces ordenar la fila por indexes de menor a mayor valor. De tal forma que
            # la fila con el valor más pequeño quede al principio.
            # axis = 0 es columnas, axis = 1 es filas.
            # TODO Probar esto y que es argsort()

            # Después de entender mejor, esto devuelve el valor mínimo de cada fila, y además,
            # los ordena de menor a mayor, y devuelve un array del siguiente tipo, por ejemplo,
            # si tuvieramos solo 2 rows: rows = [1,0]. Es to significaría que el valor mínimo
            # se encuentra en la row 1, y el siguiente valor mínimo en la row 0, y así.
            rows = D.min(axis=1).argsort()

            # Después, buscamos los valores mínimos en cada columna, y ordenándolos
            # utilizando los index de las filas calculados anteriormente
            # axis = 0 es columnas, axis = 1 es filas.

            # Después, el argmin hace lo mismo que el min, devuelve las
            # columnas con los valores mínimos de cada fila, en el caso de tener 2 filas,
            # podría devolver algo como, cols = [1, 2], indicando que en la columna 1 se encuentra
            # el valor más pequeño de la fila 0, y en la columna 2 se encuentra el valor más pequeño
            # de la columna 1. El [rows] del final sirve para ordenar los valores por filas, por tanto
            # si el valor de la fila 1, es menor que le de la fila 0, el resultado de cols será,
            # cols = [2, 1].
            cols = D.argmin(axis=1)[rows]

            # Para determinar si tenemos que actualizar, registrar o desregistrar un objeto,
            # debemos trackear cual de los indexes de las filas y columnas ya hemos examinado.
            usedRows = set()
            usedCols = set()

            # Iteramos sobre cada tupla de combinación de index (fila, columna)
            # Zip devuelve un iterador de tuplas

            # Después, el zip lo que hace es combinar las filas con las columnas, si por ejemplo las
            # rows = [1, 0], y las cols = [2, 1], entonces el zip(rows, cols) = [(1, 2), (0, 1)]
            # Esto implica, que el segundo objeto existente, hará match con el tercer input centroid,
            # ya que el zip devuelve (1, 2), y del mismo modo, el primer objeto existente hará match
            # con el segundo input centroid (0, 1).
            for (row, col) in zip(rows, cols):

                # Si ya hemos examinado la fila o la columna, la ignoramos
                if row in usedRows or col in usedCols:
                        continue

                # if the distance between centroids is greater than
                # the maximum distance, do not associate the two
                # centroids to the same object
                if D[row, col] > self.maxDistance:
                    continue

                # Si no, obtenemos el objectID de la fila actual, establecemos su
                # nuevo centroide y reseteamos el contador de desaparecido.
                objectID = objectIDs[row]
                self.objects[objectID] = inputCentroids[col]
                self.disappeared[objectID] = 0

                # Indicamos que hemos examinado el index de la fila y la columna.
                usedRows.add(row)
                usedCols.add(col)

            # Calculamos el indice de la fila y columna que todavia no hemos examinado
            unusedRows = set(range(0, D.shape[0])).difference(usedRows)
            unusedCols = set(range(0, D.shape[1])).difference(usedCols)

            # En el caso de que el numero de centroides de objeto es mayor o igual
            # al numero de nuevos centroides, debemos comprobar si algunos objetos
            # han desaparecido
            if D.shape[0] >= D.shape[1]:

                # Iterar sobre las filas sin examinar
                for row in unusedRows:

                    # Obtener el ID del objeto de la fila correspondiente
                    # e incrementar el contador de desaparecido.
                    objectID = objectIDs[row]
                    self.disappeared[objectID] += 1

                    # Revisamos que el contador no haya llegado a su límite, si es así
                    # desregistramos el objeto
                    # for warrants deregistering the object
                    if self.disappeared[objectID] > self.maxDisappeared:
                        self.deregister(objectID)

            # Si no, si el número de nuevos centroides es mayor que el número de 
            # centroides registrados, debemos registrar todos los nuevos.
            else:
                for col in unusedCols:
                    self.register(inputCentroids[col])

        # Devolvemos los objetos trackeables
        return self.objects
