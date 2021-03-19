import cv2
from matplotlib import pyplot as plt
import numpy as np
import math

# mostra a imagem na tela
def show_image(img):
    plt.imshow(img[:,:,::-1]); plt.show()

# calcula o valor do novo pixel com base na sua distancia dos pixeis originais, fazendo uma soma ponderada
def soma_ponderada(img, fator):

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

    # show_image(big_img)
    return big_img


# diminue imagem fator n multiplo de 2
# depois, interpola para tamanho original usando diversas tecnicas
def dec_int(img, fator):


    # caso fator seja ímpar, ou não inteiro
    fator = round(fator)
    if(fator%2 != 0):
        fator = fator - 1

    # cria nova imagem em branco, N vezes menor que a original (N = fator)
    h, w = round( (img.shape[0]/fator)  ), round( (img.shape[1]/fator) )

    small_img = np.zeros((h, w, 3), dtype=np.uint8)

    for i in range(h):
        for j in range(w):

            small_img[i, j] = img[i*fator, j*fator]
    
    ####################################################
    # interpola a imagem reduzida para o tamanho original
    h, w = small_img.shape[0]*fator, small_img.shape[1]*fator

    nearest = np.zeros((h, w, 3), dtype=np.uint8)

    # repete o valor mais próximo
    for i in range(h):
        for j in range(w):
            nearest[i,j] = small_img[math.floor(i/fator), math.floor(j/fator)]

    ####################################################    
    soma_ponderada = soma_ponderada(small_img, fator)

    ####################################################
    # interpolação bicubica, nativa do opencv-python
    bicubic = cv2.resize(src=small_img, dsize=(w,h), interpolation=cv2.INTER_CUBIC)

def edge_improv():

    # kernel para aguçamento das bordas
    sharp = np.array(  [[0, -1, 0],  
                        [-1, 5, -1], 
                        [0, -1, 0]])

    img = cv2.imread("test80.jpg")

    # função que realiza convolução da imagem e o kernel
    sharpened_img = cv2.filter2D(img, -1, sharp)

    show_image(sharpened_img)

# para usar a edge_improv basta descomentar a linha abaixo
# edge_improv()