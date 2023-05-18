import cv2
from numpy import where

dinho = cv2.imread(filename="../imagens/dinho.jpg")
dinho_final = dinho.copy()
gray_dinho = cv2.cvtColor(src=dinho, code=cv2.COLOR_BGR2GRAY)
dinho_head = gray_dinho[40:280, 450:600]

filtered_dinho = cv2.matchTemplate(
    image=gray_dinho,
    templ=dinho_head,
    method=cv2.TM_CCOEFF_NORMED # Correlação de Pearson normalizada
)

detection_threshold = 0.9
loc = where(filtered_dinho > detection_threshold)

pt = loc[1][0], loc[0][0]

upper_left_corner = pt
lower_right_corner = (
    upper_left_corner[0] + dinho_head.shape[1],
    upper_left_corner[1] + dinho_head.shape[0]
)
cv2.rectangle(
    img=dinho_final,
    pt1=upper_left_corner,
    pt2=lower_right_corner,
    color=(0, 0, 255), # Vermelho
    thickness=2
)

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