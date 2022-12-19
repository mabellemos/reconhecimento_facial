import face_recognition as fr
from controlador import reconhece_face, rostos_conhecidos

rosto_desconhecido = reconhece_face("./img/desconhecido.png")
if rosto_desconhecido[0]:
    foto_desconhecido = rosto_desconhecido[1][0]
    rostos, nomes = rostos_conhecidos()
    resultado = fr.compare_faces(rostos, foto_desconhecido)

    for i in range(len(rostos)):
        result = resultado[i]
        if result:
            print("\nRosto do ",nomes[i],"identificado")