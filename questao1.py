import cv2
from matplotlib import pyplot as plt
import numpy as np
import math


def show_image(name, img):

    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def dec_int(img, fator):

    #diminua imagem fator n multiplo de 2
    #interpole tamanho original repetindo pixel mais proximo

    fator = round(fator)
    if(fator%2 != 0):
        fator = fator - 1

    h, w = round( (img.shape[0]/fator)  ), round( (img.shape[1]/fator) )

    small_img = np.zeros((h, w, 3), dtype=np.uint8)

    for i in range(h):
        for j in range(w):

            small_img[i, j] = img[i*fator, j*fator]

    ####################################################
    h, w = small_img.shape[0]*fator, small_img.shape[1]*fator

    nearest = np.zeros((h, w, 3), dtype=np.uint8)

    for i in range(h):
        for j in range(w):
            nearest[i,j] = small_img[math.floor(i/fator), math.floor(j/fator)]

    ####################################################

    bicubic = cv2.resize(src=small_img, dsize=(w,h), interpolation=cv2.INTER_CUBIC)
    # plt.imshow(bicubic[:,:,::-1])
    # plt.show()

    # compare = np.hstack((nearest, bicubic))
    # plt.imshow(compare[:,:,::-1])
    # plt.show()

# def edge_improv():
#     #filtro agucamento dominio espacial melhorar qualidade subjetiva da imagem

#     laplace_kernel = np.array(  [[0, -1, 0], 
#                                 [-1, 4, -1],
#                                  [0, -1, 0]])


#     img = cv2.imread("test80.jpg")

#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#     blur = cv2.GaussianBlur(gray,(7,7), 0)
#     show_image("gray", blur)

#     edge = cv2.filter2D(blur, -1, laplace_kernel)
#     show_image("gray", edge)

#     new = img
#     for i in range(3):
#         new[:,:,i] = img[:,:,i] + edge

#     show_image("gray", new)

# edge_improv()