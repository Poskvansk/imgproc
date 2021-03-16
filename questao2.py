import cv2
from matplotlib import pyplot as plt
import numpy as np
import math


def show_image( img):

    plt.imshow(img, cmap='gray', vmin=0, vmax=255); plt.show()

    # cv2.imshow(window, img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

def gamma_correction(image_name, gamma):

    # s = c * r^gama

    img = cv2.imread(image_name)

    correct = np.zeros(img.shape, np.uint8)

    for i in range(img.shape[0]):
        correct[i] = 255 * ((img[i]/255) ** gamma)

    # correct2 = np.zeros(img.shape, np.uint8)
    # for i in range(img.shape[0]):
    #     correct2[i] = cv2.pow(img[i], gamma)

    compare = np.hstack((img, correct))
    show_image(compare)

def equalizar(image_name):
    
    img = cv2.imread(image_name)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    hist, bins = np.histogram(img.flatten(), 256, [0,256])

    plt.hist(img.flatten(),256, [0,256]); plt.show()

    # cdf = hist.cumsum()
    # cdf_normalized = cdf * float(hist.max()) / cdf.max()

    # # plt.plot(cdf); plt.show()
    # # plt.plot(cdf_normalized); plt.show()

    equalized = cv2.equalizeHist(img)

    hist_eq, bins_eq = np.histogram(equalized.flatten(), 256, [0,256])
    plt.hist(equalized.flatten(),256,[0,256]); plt.show()

    # cdf_eq = hist_eq.cumsum()
    # cdf_eq_normalized = cdf_eq * float(hist_eq.max()) / cdf_eq.max()

    # # plt.plot(cdf_eq); plt.show()
    # plt.plot(cdf_eq_normalized); plt.show()

    compare = np.hstack((img, equalized))
    show_image(compare)

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