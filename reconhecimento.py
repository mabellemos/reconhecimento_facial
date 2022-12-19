import cv2
import dlib
import numpy as np

fonte = cv2.FONT_HERSHEY_COMPLEX_SMALL

detector_faces = dlib.get_frontal_face_detector()
detector_pontos = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

imagem = cv2.imread('stallone.jpg')
faces = detector_faces(imagem, 2)

def pontos_faciais(imagem, pontosFaciais):
    for ponto in pontosFaciais.parts():
        cv2.circle(imagem, (ponto.x, ponto.y), 2, (0,255,0), 2)
    for num, coord in enumerate(pontosFaciais.parts()):
        cv2.putText(imagem, str(num), (coord.x, coord.y), fonte, .55, (0,0,255), 1)

def mascara(imagem, pontosFaciais):
    linhas = [[0, 16, False], # queixo
              [17, 21, False], # sobrancelha direita
              [22, 26, False], # sobrancelha esquerda
              [27, 30, False], # linha do nariz
              [30, 35, True], # base do nariz
              [36, 41, True], # olho esquerdo
              [42, 47, True], # olho direito
              [48, 59, True], # linha externa labio
              [60, 67, True]] # linha interna labio
    for lin in range(0, len(linhas)):
        linha = []
        for i in range(linhas[lin][0], linhas[lin][1] + 1):
            pontoxy = [pontosFaciais.part(i).x, pontosFaciais.part(i).y]
            linha.append(pontoxy)
        linha = np.array(linha)
        cv2.polylines(imagem, [linha], linhas[lin][2], (255,0,0), 2)

for face in faces:
    pontos = detector_pontos(imagem, face)
    pontos_faciais(imagem, pontos)
    mascara(imagem, pontos)
    print(pontos.parts())

cv2.imshow('Faces', imagem)
cv2.waitKey(0)
cv2.destroyAllWindows()