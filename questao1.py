import cv2
from matplotlib import pyplot as plt
import numpy as np
import math


def show_image(img):

    plt.imshow(img[:,:,::-1]); plt.show()


def media_valores(img, fator):

    h, w = img.shape[0]*fator, img.shape[1]*fator



    big_img = np.zeros((h,w,3), dtype=np.uint8)


    for i in range(img.shape[0]):
        for j in range(img.shape[1]):

            big_img[i* fator, j* fator] = img[i,j]


    for i in range(h):
        for j in range(w):

            dist = (i/fator) - math.floor(i/fator)
            w1 = 1-dist
            w2 = 1-w1

            big_img[i, j] = (img[ math.floor(i/fator) ,math.floor(j/fator)]) *w1 + (img[math.ceil(i/fator)-1, math.ceil(j/fator)-1]) *w2

    show_image(big_img)


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

    media_valores(small_img, fator)


def edge_improv():

    # sharp = np.array(  [[0, -1, 0],  
    #                     [-1, 5, -1], 
    #                     [0, -1, 0]])

    laplace = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])


    img = cv2.imread("test80.jpg")

    # sharpened_img = cv2.filter2D(img, -1, sharp)

    blur = cv2.GaussianBlur(img, (3,3), sigmaX=1)

    img2 = cv2.filter2D(img, -1, laplace)

    show_image("sharp", img2)
