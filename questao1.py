import cv2
from matplotlib import pyplot as plt
import numpy as np
import math


def dec_int(img, fator):

    #diminua imagem fator n multiplo de 2
    #interpole tamanho original repetindo pixel mais proximo

    if(fator%2 != 0):
        fator -= 1

    h, w = int(img.shape[0]/fator), int(img.shape[1]/fator)

    small_img = np.zeros((h, w, 3), dtype=np.uint8)

    for i in range(h):
        for j in range(w):

            small_img[i, j] = img[i*fator, j*fator]

        
    plt.imshow(small_img[:,:,::-1])
    plt.show()


def edge_improv():
    #filtro agucamento dominio espacial melhorar qualidade subjetiva da imagem
    a=2
    return
