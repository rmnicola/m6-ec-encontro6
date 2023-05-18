import cv2

# Carrega os classificadores em cascata
face_cascade = cv2.CascadeClassifier(
    filename=f"{cv2.data.haarcascades}/haarcascade_frontalface_default.xml"
)
dinho = cv2.imread(filename="../imagens/dinho.jpg")
gray_dinho = cv2.cvtColor(src=dinho, code=cv2.COLOR_BGR2GRAY)
# Passa o detector em cascata pelo método multi scale
# Esse método vai diminuindo a imagem a cada passada
# e testa o classificador em sequência
faces = face_cascade.detectMultiScale(
    image=gray_dinho, 
    scaleFactor=1.05, # Mudança de escala a cada passada
    minNeighbors=5 # Verifica os vizinhos antes de promover o ponto a ret
)
# Pega os dados do primeiro retangulo encontrado
x, y, w, h = faces[0]
# Desenha o retangulo
cv2.rectangle(
    img=dinho,
    pt1=(x, y),
    pt2=(x+w, y+h),
    color=(0,0,255),
    thickness=2
)

# Exibe tudo
cv2.imshow('Dinho', dinho)
cv2.waitKey(0)
cv2.destroyAllWindows()