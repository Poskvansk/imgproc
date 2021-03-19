import cv2
from matplotlib import pyplot as plt
import numpy as np
import math


# mostra imagem em nivel de cinza
def show_image( img):

    plt.imshow(img, cmap='gray', vmin=0, vmax=255); plt.show()


# correção gama
def gamma_correction(image_name, gamma):

    img = cv2.imread(image_name)

    # valores dos pixeis de [0. , 1.]
    img = img/255.0

    # eleva todos os pixeis da imagem à gama
    correct = cv2.pow(img, gamma)
    correct = np.uint8(correct*255)

    show_image(correct)


# Função para equalizar histograma
# utiliza equalizeHist do opencv
def equalizar(image_name):
    
    img = cv2.imread(image_name)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    hist, bins = np.histogram(img.flatten(), 256, [0,256])
    # plt.hist(img.flatten(),256, [0,256]); plt.show()

    cdf = hist.cumsum()
    plt.plot(cdf); plt.show()

    equalized = cv2.equalizeHist(img)

    hist_eq, bins_eq = np.histogram(equalized.flatten(), 256, [0,256])
    # plt.hist(equalized.flatten(),256,[0,256]); plt.show()

    cdf_eq = hist_eq.cumsum()
    plt.plot(cdf_eq); plt.show()



# funcao principal que chama as outras funcoes
# as chamadas estão comentadas, basta descomentar e executar o programa
def questao2():

    # gamma_correction("car.png", 0.5)
    # gamma_correction("car.png", 0.75)
    # gamma_correction("car.png", 1.5)
    # gamma_correction("car.png", 2)

    # gamma_correction("crowd.png", 0.5)
    # gamma_correction("crowd.png", 0.75)
    # gamma_correction("crowd.png", 1.5)
    # gamma_correction("crowd.png", 2)

    # gamma_correction("university.png", 0.5)
    # gamma_correction("university.png", 0.75)
    # gamma_correction("university.png", 1.5)
    # gamma_correction("university.png", 2)
    ##################################################################

    equalizar("car.png")
    # equalizar("crowd.png")
    # equalizar("university.png")

questao2()