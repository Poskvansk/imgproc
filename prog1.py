from questao1 import *

def main():

    img = cv2.imread('test80.jpg')

    # plt.imshow(img[:,:,::-1])
    # plt.show()

    fator = 2
    dec_int(img, fator)

main()    