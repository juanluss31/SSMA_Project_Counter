# coding=utf-8
from centroidTrackable.centroidtracker import CentroidTracker
from centroidTrackable.trackableobject import TrackableObject
from imutils.video import VideoStream
from imutils.video import FPS
from dotenv import load_dotenv
import requests
import datetime
import os
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2
import ffmpeg

# Creamos los argumentos
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", required=True, help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True, help="path to Caffe pre-trained model")
ap.add_argument("-i", "--input", type=str, help="path to optional input video file")
ap.add_argument("-o", "--output", type=str, help="path to optional output video file")
ap.add_argument("-c", "--confidence", type=float, default=0.4, help="minimum probability to filter weak detections")
ap.add_argument("-s", "--skip-frames", type=int, default=30, help="# of skip frames between detections")
ap.add_argument("-t", "--test", type=int, default=0,
                help="1 or 0 (default). If 0, it will connect with the database. If 1, it wont connect to the database, its for testing purposes.")
args = vars(ap.parse_args())

# initialize the list of class labels MobileNet SSD was trained to
# detect
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

# Cargamos nuestro modelo serializado
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

newframetime=0
prevframetime=0

# Si no se especifica una ruta de input video, utilizaremos la webcam
if not args.get("input", False):
    print("[INFO] starting video stream...")
    vs = VideoStream(src=0, resolution=(320,240)).start()
    time.sleep(2.0)
# Si se especiifica, entonces lo cargamos
else:
    print("[INFO] opening video file...")
    vs = cv2.VideoCapture(args["input"])

# Inicializamos el creador de video
writer = None

# Inicializamos las dimensiones de frame
W = None
H = None

# Instaciamos nuestro centroid tracker, e inicializamos una lista para almacenar
# cada uno de nuestros trackers de correlación de dlib, seguidos de un diccionario
# para mapear cada objectId unica a un TrackableObject
ct = CentroidTracker(maxDisappeared=40, maxDistance=50)
trackers = []
trackableObjects = {}

# Inicializamos el numero total de frames procesado hasta ahora, junto con el
# numero total de objetos que se han movido la derecha o la izquierda
totalFrames = 0
totalRight = 0
totalLeft = 0
header = {}

startTime = datetime.datetime.now()

mode = 0

if args["test"] == 0:
    load_dotenv()
    print("[INFO] estas en modo conexion")
    # response = requests.post(os.getenv('URLLOGIN'), data={'username': os.getenv('USERNAME'), 'password': os.getenv('PASS')})
    # print(response.text)
    headers = {'Authorization': 'Basic YWRtaW46YWRtaW4='}
else:
    mode = 1
    print("[INFO] estas en modo test")

# Ejecutamos el estimador de frames por segundo
fps = FPS().start()

# Define the RTMP URL
rtmp_url = "rtmp://localhost/live/counter"

# Initialize the FFmpeg stream
process = None

# Define the desired FPS
desired_fps = 30
frame_time = 1 / desired_fps  # Time per frame in seconds

# Bucle principal
while True:
    start_time = time.time()  # Record the start time of the frame

    # Obtenemos el siguiente frame y lo procesamos (tanto si usamos camaro como video)
    frame = vs.read()
    frame = frame[1] if args.get("input", False) else frame

    # Si estamos viendo un video y no tomamos un frame, entonces hemos acabado el video
    if args["input"] is not None and frame is None:
        break

    # Hacemos un resize al frame para tener un ancho máximo de 500px (menos datos, más
    # rápido se procesa), luego convertimos el frame de BGR a RGB para dlib
    frame = imutils.resize(frame, width=240)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Si las dimensiones del frame estan vacias, las asignamos
    if W is None or H is None:
        (H, W) = frame.shape[:2]
        # Initialize the FFmpeg stream once dimensions are known
        process = (
            ffmpeg
            .input('pipe:', format='rawvideo', pix_fmt='bgr24', s=f"{W}x{H}", framerate=30)
            .output(
                rtmp_url,
                vcodec='libx264',
                preset='ultrafast',  # Use ultrafast preset to reduce latency
                tune='zerolatency',
                pix_fmt='yuv420p',  # Ensure compatibility with RTMP
                g=30,  # Set GOP size to match the frame rate
                format='flv'
            )
            .run_async(pipe_stdin=True)
        )

    # Si estamos creando un video, inicializamos el video writer
    if args["output"] is not None and writer is None:
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(args["output"], fourcc, 30,
            (W, H), True)

    # Inicializamos el estado actual, junto con nuestra lista de bounding boxes retornada 
    # por nuestro detector de objetos o por los trackers de correlacion de dlib
    # El estado puede ser esperando, detectando, tracking
    status = "Waiting"
    rects = []

    # Comprobar si deberíamos utilizar un método de detección de objetos más costoso
    # computacionalmente para ayudar a nuestro tracker
    if totalFrames % args["skip_frames"] == 0:
        # Cambiar el estado e inicializar nuestro nuevo conjunto de trackers
        status = "Detecting"
        trackers = []

        # Convertir el frame a un blob y pasar el blob a través de la network para obtener las detecciones
        blob = cv2.dnn.blobFromImage(frame, 0.007843, (W, H), 127.5)
        net.setInput(blob)
        detections = net.forward()

        # loop sobre las detecciones
        for i in np.arange(0, detections.shape[2]):
            # Extraemos la confianza o probabilidad asociada a la prediccion
            confidence = detections[0, 0, i, 2]

            # Filtramos las detecciones debiles
            if confidence > args["confidence"]:
                # Extraemos el índice de las detecciones
                idx = int(detections[0, 0, i, 1])

                # Si no es una persona lo ignoramos
                if CLASSES[idx] != "person":
                    continue

                # Calculamos las coordenadas (x, y) de las bounding box del objeto
                box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
                (startX, startY, endX, endY) = box.astype("int")

                # Construimos un rectangulo de dlib, utilizando las coordenadas de la
                # bounding box, y empezamos el tracker de correlacion de dlib
                tracker = dlib.correlation_tracker()
                rect = dlib.rectangle(startX, startY, endX, endY)
                tracker.start_track(rgb, rect)

                # Añadimos el tracker a nuestra lista de tracker para utilizarlos cuando
                # hagamso skip de frames
                trackers.append(tracker)

    # Si no, debemos utilizar nuestro objeto "trackers" en vez de el objeto "detectores" para
    # obtener un mayor procesamiento de frames
    else:
        # Loop sobre los trakers
        for tracker in trackers:
            # Cambia el status del sistema a "tracking" en luegar de "waiting" o "detecting"
            status = "Tracking"

            # Actualizamos el tracker y obtenemos la posicion actualizada
            tracker.update(rgb)
            pos = tracker.get_position()

            # Desglosamos el objeto de posicion
            startX = int(pos.left())
            startY = int(pos.top())
            endX = int(pos.right())
            endY = int(pos.bottom())

            # Añadimos las coordenadas de la bounding box a la lista de rectangulos
            rects.append((startX, startY, endX, endY))

    # Dibujamos una linea vertical en el centro del frame. Una vez un objeto cruce esta linea,
    # se determinara si se esta moviendo hacia la derecha o hacia la izquierda
    # cv2.line(frame, (0, H // 2), (W, H // 2), (0, 255, 255), 2)
    cv2.line(frame, (W // 2, 0), (W // 2, H), (255, 0, 255), 2)

    # Usamos el centroid tracker para asociar el antiguo centroide del objeto con los nuevos computados
    objects = ct.update(rects)

    # Loop sobre los objetos trackeados
    for (objectID, centroid) in objects.items():
        # Comprobamos si ya existe un objeto trackeado con el objectID acutal
        to = trackableObjects.get(objectID, None)

        # Si no hay ningun objeto trackeable, creamos uno
        if to is None:
            to = TrackableObject(objectID, centroid)

        # Si nom ya existe un objeto trackeable asi que podemos utilizarlos para calcular la direccion
        else:
            # La diferencia entre la coordenada x del centroide actual, y la media de los centroides anteriores
            # nos indicará en qué dirección se está moviendo el objeto (positivo derecha, negativo izquierda)
            x = [c[0] for c in to.centroids]
            directionX = centroid[0] - np.mean(x)

            # Comprobar si el objeto ha sido contado o no
            if not to.counted:
                if directionX > 0 and centroid[0] > W // 2:  # Moviéndose hacia la derecha
                    totalRight += 1
                    to.counted = True
                elif directionX < 0 and centroid[0] < W // 2:  # Moviéndose hacia la izquierda
                    totalLeft += 1
                    to.counted = True

        # Almacenamos los objectos trackeables en el diccionario
        trackableObjects[objectID] = to

        # Dibujamos el Id del objeto y el centroide del mismo con el output frame
        text = "ID {}".format(objectID)
        cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

    # Construimos una tupla de informacion que mostraremos en el frame
    info = [
        ("Right", totalRight),
        ("Left", totalLeft),
        ("Status", status),
    ]

    # Loop sobre las tuplas de informacion y dibujarlas en el frame
    for (i, (k, v)) in enumerate(info):
        text = "{}: {}".format(k, v)
        cv2.putText(frame, text, (10, H - ((i * 20) + 20)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # Comprobamos si debemos crear un video
    if writer is not None:
        writer.write(frame)

    # CALCULO DE FPS
    newframetime = time.time()
    framespersecond = 1/(newframetime-prevframetime)
    prevframetime = newframetime

    framespersecond = int(framespersecond)
    framespersecond = str(framespersecond)

    cv2.putText(frame, framespersecond, (7, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 255, 0), 3, cv2.LINE_AA)

    # Mostramos el output frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # Write the frame to FFmpeg's stdin for streaming
    if process.stdin:
        process.stdin.write(frame.tobytes())

    # Si se presiona la tecla q acabamos
    if key == ord("q"):
        break

    # Incrementamos el numero total de frames procesados hasta ahora y actualizamos el contador de fps
    totalFrames += 1
    fps.update()

    # Enforce a stable FPS
    elapsed_time = time.time() - start_time
    delay = max(0, frame_time - elapsed_time)  # Calculate the remaining time for the frame
    time.sleep(delay)  # Introduce a delay to maintain the desired FPS

    if mode == 0 and int((datetime.datetime.now() - startTime).total_seconds()) > 5:
        startTime = datetime.datetime.now()
        # print("[INFO] sending data to backend")
        # Log the payload being sent
        payload = {'right': totalRight, 'left': totalLeft}
        # print(f"[DEBUG] Payload: {payload}")
        x = requests.post(os.getenv('URLUPDATE'), json=payload, headers=headers)
        # print(f"[DEBUG] Server Response: {x.status_code}, {x.text}")

if mode == 0:
    x = requests.post(os.getenv('URLUPDATE'), json={'right': totalRight, 'left': totalLeft}, headers=headers)
    print(x)

# Clean up the FFmpeg process
if process.stdin:
    process.stdin.close()
process.wait()

# Terminamos el timer y mostramos la informacion de FPS
fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# Mostramos los contadores en el terminal
print("[INFO] personas hacia la derecha: " + str(totalRight))
print("[INFO] personas hacia la izquierda: " + str(totalLeft))
print("[INFO] personas en total: " + str(totalRight - totalLeft))

# Comprobamos si necesitamos terminar con el video
if writer is not None:
    writer.release()

# Si no estamos utilizando un archivo de video, apagamos la webcam
if not args.get("input", False):
    vs.stop()

    # Si si estamos utilizando un archivo de video lo cerramos
else:
    vs.release()

# Cerramos las ventanas abiertas
cv2.destroyAllWindows()