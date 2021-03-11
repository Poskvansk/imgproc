# import cv2
# from matplotlib import pyplot as plt
# import numpy as np
# import math

from questao1 import *

def main():

    #########################################
    ## Questao 1

    img = cv2.imread('crowd.png')

    # plt.imshow(img[:,:,::-1])
    # plt.show()

    fator = 3
    dec_int(img, fator)


main()    