import cv2
from matplotlib import pyplot as plt
import numpy as np
import math


def show_image(window, img):

    # plt.imshow(matrix, cmap=plt.get_cmap('gray'))

    cv2.imshow(window, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def gamma_correction(image_name, gamma):

    # # s = c r^gama
    img = cv2.imread(image_name)

    h, w = img.shape[0], img.shape[1]

    for i in range(h):
        for j in range(w):

            img[i, j] = 255 * ((img[i, j]/255) ** gamma)

    img2 = np.array( 255*( img/255 ) ** gamma, dtype = 'uint8')

    compare = np.hstack((img, img2))

    show_image("compare", compare)

def equalizar(image_name):
    
    img = cv2.imread(image_name)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    hist, bins = np.histogram(img.flatten(), 256, [0,256])

    # plt.hist(img.flatten(),256,[0,256]); plt.show()

    cdf = hist.cumsum()
    cdf_normalized = cdf * float(hist.max()) / cdf.max()

    # plt.plot(cdf); plt.show()
    # plt.plot(cdf_normalized); plt.show()

    equalized = cv2.equalizeHist(img)

    hist_eq, bins_eq = np.histogram(equalized.flatten(), 256, [0,256])
    plt.hist(equalized.flatten(),256,[0,256]); plt.show()

    cdf_eq = hist_eq.cumsum()
    cdf_eq_normalized = cdf_eq * float(hist_eq.max()) / cdf_eq.max()

    # plt.plot(cdf_eq); plt.show()
    plt.plot(cdf_eq_normalized); plt.show()

    compare = np.hstack((img, equalized))
    show_image("compare", compare)

def questao2():

    # gamma_correction("car.png", 1.5)

    ##################################################################

    # equalizar("car.png")
    # equalizar("crowd.png")
    # equalizar("university.png")

questao2()