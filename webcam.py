import numpy as np
import face_recognition as fr
import cv2
from controlador import rostos_conhecidos

rostos, nomes = rostos_conhecidos()

webcam = cv2.VideoCapture(0)
while True:
    verificador, print = webcam.read()
    rgb_foto = print[:, :, ::-1]

    face_localizaco = fr.face_locations(rgb_foto)
    face_encodings = fr.face_encodings(rgb_foto, face_localizaco)

    for (emcima, direita, embaixo, esquerda), face_encodings in zip(face_localizaco, face_encodings):
        resultado = fr.compare_faces(rostos, face_encodings)

        distancia_faces = fr.face_distance(rostos, face_encodings)

        melhor_id = np.argmin(distancia_faces)
        if resultado[melhor_id]:
            nome = nomes[melhor_id]
        else:
            print("\nDesconhecido")
        
        cv2.rectangle(print, (esquerda, emcima), (direita, embaixo), (0, 0, 255), 2)

        cv2.rectangle(print, (esquerda, embaixo -35), (direita, emcima), (0, 0, 255), cv2.FILLED)
        fonte_texto = cv2.FONT_HERSHEY_COMPLEX_SMALL
        cv2.putText(print, nome, (esquerda + 6, embaixo - 6), fonte_texto, 1.0, (255, 255, 255), 1)

        cv2.imshow("Reconhecimento facial", print)
    if cv2.waitKey(5) == 27:
        break

webcam.release()
cv2.destroyAllWindows()