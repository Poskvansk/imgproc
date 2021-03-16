
from questao1 import *

def main():

    #########################################
    ## Questao 1

    img = cv2.imread('test80.jpg')

    # plt.imshow(img[:,:,::-1])
    # plt.show()

    fator = 4
    dec_int(img, fator)


main()    