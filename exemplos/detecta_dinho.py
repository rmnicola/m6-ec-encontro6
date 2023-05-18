import cv2
import numpy as np

dinho = cv2.imread(filename="../imagens/dinho.jpg")
# Copia da imagem original para desenhar o retangulo
dinho_final = dinho.copy()
gray_dinho = cv2.cvtColor(src=dinho, code=cv2.COLOR_BGR2GRAY)
# Crop da cabeça do dinho
dinho_head = gray_dinho[40:280, 450:600]

# Filtro baseado em correlação
filtered_dinho = cv2.matchTemplate(
    image=gray_dinho,
    templ=dinho_head,
    method=cv2.TM_CCOEFF_NORMED # Correlação de Pearson normalizada
)
# Valores maiores que o threshold são considerados como
# se o objeto tenha sido encontrado naquele pixel
detection_threshold = 0.9
# Encontra os pontos em que o valor passa do threshold
pontos = np.where(filtered_dinho > detection_threshold)
# Escolhe o primeiro ponto do conjunto
pt = pontos[1][0], pontos[0][0]

# Define os pontos do retangulo
upper_left_corner = pt
lower_right_corner = (
    upper_left_corner[0] + dinho_head.shape[1],
    upper_left_corner[1] + dinho_head.shape[0]
)
# Desenha o retangulo
cv2.rectangle(
    img=dinho_final,
    pt1=upper_left_corner,
    pt2=lower_right_corner,
    color=(0, 0, 255), # Vermelho
    thickness=2
)

# Exibe tudo
cv2.imshow(winname="Dinho", mat=dinho)
cv2.waitKey(delay=0)
cv2.destroyWindow(winname="Dinho")

cv2.imshow(winname="Gray Dinho", mat=gray_dinho)
cv2.waitKey(delay=0)
cv2.destroyWindow(winname="Gray Dinho")

cv2.imshow(winname="Mask", mat=dinho_head)
cv2.waitKey(delay=0)
cv2.destroyWindow(winname="Mask")

cv2.imshow(winname="Filtered Dinho", mat=filtered_dinho)
cv2.waitKey(delay=0)
cv2.destroyWindow(winname="Filtered Dinho")

cv2.imshow(winname="Final Dinho", mat=dinho_final)
cv2.waitKey(delay=0)
cv2.destroyWindow(winname="Final Dinho")