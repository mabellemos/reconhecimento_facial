import face_recognition as fr

def reconhece_face(url_foto):
    foto = fr.load_image_file(url_foto)
    rostos = fr.face_encondings(foto)
    if len(rostos) > 0:
        return True, rostos
    return False, []

def rostos_conhecidos():
    rostos = []
    nomes = []

    foto1 = reconhece_face('./img/maria.jpg')
    if foto1[0]:
        rostos.append(foto1[1][0])
        nomes.append("Maria")

    foto2 = reconhece_face('./img/izabel.jpg')
    if foto2[0]:
        rostos.append(foto2[1][0])
        nomes.append("Izabel")
    return rostos, nomes