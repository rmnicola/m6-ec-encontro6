import cv2

face_cascade = cv2.CascadeClassifier(
    filename=cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)
dinho = cv2.imread(filename="../imagens/dinho2.jpg")
gray_dinho = cv2.cvtColor(src=dinho, code=cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(
    image=gray_dinho, 
    scaleFactor=1.3, 
    minNeighbors=5
)

x, y, w, h = faces[0]
cv2.rectangle(
    img=dinho,
    pt1=(x, y),
    pt2=(x+w, y+h),
    color=(0,0,255),
    thickness=2
)

cv2.imshow('Dinho', dinho)
cv2.waitKey(0)
cv2.destroyAllWindows()