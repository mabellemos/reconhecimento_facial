import os
import glob
import _pickle as cPickle
import dlib
import cv2
import numpy as np

detector_faces = dlib.get_frontal_face_detector()
detector_pontos = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
reconhecimento_facial = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")


indices = np.load('indices.pickle', allow_pickle=True)
descritores = np.load('descritores.npy')
limiar = 0.5

for data in glob.glob(os.path.join("img_teste", "*.jpg")):
    imagem = cv2.imread(data)
    faces = detector_faces(imagem, 1)
    for face in faces:
        e, t, d, b = (int(face.left()),
                      int(face.top()),
                      int(face.right()),
                      int(face.bottom()))
        pontos = detector_pontos(imagem, face)
        descritor = reconhecimento_facial.compute_face_descriptor(imagem,
                                                                  pontos)
        lista_caracteristicas = [fd for fd in descritor]
        array_descritor = np.asarray(lista_caracteristicas,
                                     dtype=np.float64)
        array_descritor = array_descritor[np.newaxis, :]

        pontos_coincidentes = np.linalg.norm(array_descritor - descritores,
                                             axis=1)
        print(f'Margem de Coincidência: {pontos_coincidentes}')
        minimo_permitido = np.argmin(pontos_coincidentes)
        print(minimo_permitido)
        margem_minima_permitida = pontos_coincidentes[minimo_permitido]
        print(margem_minima_permitida)

        if margem_minima_permitida <= limiar:
            nome = os.path.split(indices[minimo_permitido])[1].split(".")[0]
        else:
            nome = ' '

        cv2.rectangle(imagem, (e, t), (d, b), (0, 255, 255), 2)
        texto = "{} {:.4f}".format(nome, margem_minima_permitida)
        cv2.putText(imagem, texto, (d, t), cv2.FONT_HERSHEY_TRIPLEX, 0.7, (0, 255, 255))

    cv2.imshow("Face Detectada", imagem)
    cv2.waitKey(0)

cv2.destroyAllWindows()