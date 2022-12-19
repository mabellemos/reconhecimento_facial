import cv2
import mediapipe as mp

webcam = cv2.VideoCapture(0)
solucao_reconhecimento_rosto = mp.solutions.face_detection
reconhecedor_rosto = solucao_reconhecimento_rosto.FaceDetection()
desenho = mp.solutions.drawing_utils

while True:
    verificador, frame = webcam.read()
    if not verificador:
        break
    listas_rostos = reconhecedor_rosto.process(frame)

    if listas_rostos.detections:
        for rostos in listas_rostos.detections:
            desenho.draw_detection(frame, rostos)
    cv2.imshow("Reconhecimento facial", frame)
    if frame == "":
        print("reconhecido")
    if cv2.waitKey(5) == 27:
        break

webcam.release()
cv2.destroyAllWindows()