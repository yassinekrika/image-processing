import random
import cv2
import math
from PIL import Image, ImageTk
import numpy as np
import copy
from pick import pick


imageColor = cv2.imread('/home/yassg4mer/Project/tp_tmn/partie2/m.bmp')
img= cv2.cvtColor(imageColor, cv2.COLOR_RGB2GRAY)
cv2.imwrite('/home/yassg4mer/Project/tp_tmn/partie2/gray.jpg', img)
ImageBruitGaussine= copy.copy(img)
ImageBruitProiveEtSel= copy.copy(img)

def CaluculerKennelGaussin(tita, k):
    TitaEntier = math.ceil(3 * tita)  # l'entier le plus proche
    # k=2*TitaEntier+1 # k est le nombre des colons et ligne de kennel
    masque = np.zeros((k, k))
    for i in range(k):

        for j in range(k):
            y = i - math.ceil(k / 2) - 1
            x = j - math.ceil(k / 2) - 1
            masque[i, j] = (1 / (2 * math.pi * tita * tita)) * math.exp(-(x * x + y * y) / (2 * tita * tita))

    return masque

def BruitGaussine():
    sigma = float(input('set sigma value : '))
    TitaEntier = math.ceil(3 * sigma)
    taille = 2 * TitaEntier + 1
    Xtest = copy.copy(img)
    kennel = CaluculerKennelGaussin(sigma, 3)
    k = len(kennel)
    height, width =  Xtest.shape[:2]
    img1 = copy.copy( Xtest)
    for i in range(height):
        for j in range(width):
            # if (i - 1 + k >  1 and j - 1 + k > 1 and i - 1 + k < height - 1 and j - 1 + k  < width - 1):
            Xtest[i, j] =  Xtest[i, j] + Calculer_valaur_de_Pixel_kennel(kennel, img1, 3, i, j)

    X2= cv2.GaussianBlur(img, (3, 3), sigma)
    cv2.imwrite("photo_G.jpg", img)

def BruiteProiveEtSel():
    proba = 0.05
    height, width = img.shape[:2]
    global ImageBruitProiveEtSel
    for k in range(int(height * width * proba)):
        i = random.randint(0, height - 1)
        j = random.randint(0, width - 1)
        choixcolor = random.randint(0, 1)
        if (choixcolor < 0.5):
            ImageBruitProiveEtSel[i, j] = 0
        else:
            ImageBruitProiveEtSel[i, j] = 255
    cv2.imwrite("/home/yassg4mer/Project/tp_tmn/partie2/photo_BPS1.jpg", ImageBruitProiveEtSel)

def Calculer_valaur_de_Pixel_kennel(kennel, img, div, i, j):
    s = 0
    t=len(kennel)

    for k in range(len(kennel)):
        for l in range(len(kennel)):
            s = s + img[i + k, j + l] * kennel[k, l]

    return s

def prewitt(img):
    taille=3
    height, width = img.shape[:2]

    horizontal = np.array([[1., 0., -1.],
                          [1., 0., -1.],
                          [1., 0., -1.]])

    vertical = np.array([[1., 1., 1.],
                          [0., 0., 0.],
                          [-1., -1., -1.]])

    newgradientImage = np.zeros((height, width))
    newVerticalImage = np.zeros((height, width))
    newHorisontalImage = np.zeros((height, width))

    for i in range(1, height - 1):
        for j in range(1, width - 1):
            horizontalGrad = (horizontal[0, 0] * img[i - 1, j - 1]) + \
                            (horizontal[0, 1] * img[i - 1, j]) + \
                            (horizontal[0, 2] * img[i - 1, j + 1]) + \
                            (horizontal[1, 0] * img[i, j - 1]) + \
                            (horizontal[1, 1] * img[i, j]) + \
                            (horizontal[1, 2] * img[i, j + 1]) + \
                            (horizontal[2, 0] * img[i + 1, j - 1]) + \
                            (horizontal[2, 1] * img[i + 1, j]) + \
                            (horizontal[2, 2] * img[i + 1, j + 1])

            verticalGrad = (vertical[0, 0] * img[i - 1, j - 1]) + \
                        (vertical[0, 1] * img[i - 1, j]) + \
                        (vertical[0, 2] * img[i - 1, j + 1]) + \
                        (vertical[1, 0] * img[i, j - 1]) + \
                        (vertical[1, 1] * img[i, j]) + \
                        (vertical[1, 2] * img[i, j + 1]) + \
                        (vertical[2, 0] * img[i + 1, j - 1]) + \
                        (vertical[2, 1] * img[i + 1, j]) + \
                        (vertical[2, 2] * img[i + 1, j + 1])

            mag = np.sqrt(pow(horizontalGrad, 2.0) + pow(verticalGrad, 2.0))
            newgradientImage[i - 1, j - 1] = mag
            newHorisontalImage[i - 1, j - 1] = horizontalGrad
            newVerticalImage[i - 1, j - 1] = verticalGrad

    cv2.imwrite("/home/yassg4mer/Project/tp_tmn/partie2/PrewittX.jpg", newHorisontalImage)
    cv2.imwrite("/home/yassg4mer/Project/tp_tmn/partie2/PrewittY.jpg", newHorisontalImage)
    cv2.imwrite("/home/yassg4mer/Project/tp_tmn/partie2/Prewitt.jpg", newgradientImage)

def roberts(img):
    height, width = img.shape[:2]   
    horizontal = np.array([[0, 0, -1.],
                          [0, 1., 0],
                          [0, 0, 0]])

    vertical = np.array([[-1., 0, 0],
                          [0, 1., 0],
                          [0, 0, 0]])
    newhorizontalImage = np.zeros((height, width))
    newverticalImage = np.zeros((height, width))
    newgradientImage = np.zeros((height, width))

    for i in range(1, height - 1):
        for j in range(1, width - 1):
            horizontalGrad = (horizontal[0, 0] * img[i - 1, j - 1]) + \
                            (horizontal[0, 1] * img[i - 1, j]) + \
                            (horizontal[0, 2] * img[i - 1, j + 1]) + \
                            (horizontal[1, 0] * img[i, j - 1]) + \
                            (horizontal[1, 1] * img[i, j]) + \
                            (horizontal[1, 2] * img[i, j + 1]) + \
                            (horizontal[2, 0] * img[i + 1, j - 1]) + \
                            (horizontal[2, 1] * img[i + 1, j]) + \
                            (horizontal[2, 2] * img[i + 1, j + 1])

            newhorizontalImage[i - 1, j - 1] = abs(horizontalGrad)

            verticalGrad = (vertical[0, 0] * img[i - 1, j - 1]) + \
                        (vertical[0, 1] * img[i - 1, j]) + \
                        (vertical[0, 2] * img[i - 1, j + 1]) + \
                        (vertical[1, 0] * img[i, j - 1]) + \
                        (vertical[1, 1] * img[i, j]) + \
                        (vertical[1, 2] * img[i, j + 1]) + \
                        (vertical[2, 0] * img[i + 1, j - 1]) + \
                        (vertical[2, 1] * img[i + 1, j]) + \
                        (vertical[2, 2] * img[i + 1, j + 1])

            newverticalImage[i - 1, j - 1] = abs(verticalGrad)

            mag = np.sqrt(pow(horizontalGrad, 2.0) + pow(verticalGrad, 2.0))
            newgradientImage[i - 1, j - 1] = mag

    cv2.imwrite("/home/yassg4mer/Project/tp_tmn/partie2/RobertsX.jpg", newhorizontalImage)
    cv2.imwrite("/home/yassg4mer/Project/tp_tmn/partie2/RobertsY.jpg", newverticalImage)
    cv2.imwrite("/home/yassg4mer/Project/tp_tmn/partie2/Roberts.jpg", newgradientImage)

def gradient(img):
    height, width = img.shape[:2]   
    horizontal = np.array([[-1, 1]])

    vertical = np.array([[1],[-1]])
    newhorizontalImage = np.zeros((height, width))
    newverticalImage = np.zeros((height, width))
    newgradientImage = np.zeros((height, width))

    for i in range(1, height - 1):
        for j in range(1, width - 1):
            horizontalGrad = (horizontal[0, 0] * img[i - 1, j - 1]) + \
                            (horizontal[0, 1] * img[i - 1, j]) + \
                            (horizontal[0, 2] * img[i - 1, j + 1]) + \
                            (horizontal[1, 0] * img[i, j - 1]) + \
                            (horizontal[1, 1] * img[i, j]) + \
                            (horizontal[1, 2] * img[i, j + 1]) + \
                            (horizontal[2, 0] * img[i + 1, j - 1]) + \
                            (horizontal[2, 1] * img[i + 1, j]) + \
                            (horizontal[2, 2] * img[i + 1, j + 1])

            newhorizontalImage[i - 1, j - 1] = abs(horizontalGrad)

            verticalGrad = (vertical[0, 0] * img[i - 1, j - 1]) + \
                        (vertical[0, 1] * img[i - 1, j]) + \
                        (vertical[0, 2] * img[i - 1, j + 1]) + \
                        (vertical[1, 0] * img[i, j - 1]) + \
                        (vertical[1, 1] * img[i, j]) + \
                        (vertical[1, 2] * img[i, j + 1]) + \
                        (vertical[2, 0] * img[i + 1, j - 1]) + \
                        (vertical[2, 1] * img[i + 1, j]) + \
                        (vertical[2, 2] * img[i + 1, j + 1])

            newverticalImage[i - 1, j - 1] = abs(verticalGrad)

            mag = np.sqrt(pow(horizontalGrad, 2.0) + pow(verticalGrad, 2.0))
            newgradientImage[i - 1, j - 1] = mag

    cv2.imwrite("/home/yassg4mer/Project/tp_tmn/partie2/GradientX.jpg", newhorizontalImage)
    cv2.imwrite("/home/yassg4mer/Project/tp_tmn/partie2/GradientY.jpg", newverticalImage)
    cv2.imwrite("/home/yassg4mer/Project/tp_tmn/partie2/Gradient.jpg", newgradientImage)

def sobel(img):
    height, width = img.shape[:2]

    vertical = np.array([[-2., -3, -2.],
                        [0., 0, 0.],
                        [2., 3, 2.]])

    horizontal = np.array([[-2., 0., 2.],
                        [-3, 0, 3],
                        [-2., 0., 2.]])

    newhorizontalImage = np.zeros((height, width))
    newverticalImage = np.zeros((height, width))
    newgradientImage = np.zeros((height, width))

    for i in range(1, height - 1):
        for j in range(1, width - 1):
            horizontalGrad = (horizontal[0, 0] * img[i - 1, j - 1]) + \
                            (horizontal[0, 1] * img[i - 1, j]) + \
                            (horizontal[0, 2] * img[i - 1, j + 1]) + \
                            (horizontal[1, 0] * img[i, j - 1]) + \
                            (horizontal[1, 1] * img[i, j]) + \
                            (horizontal[1, 2] * img[i, j + 1]) + \
                            (horizontal[2, 0] * img[i + 1, j - 1]) + \
                            (horizontal[2, 1] * img[i + 1, j]) + \
                            (horizontal[2, 2] * img[i + 1, j + 1])

            newhorizontalImage[i - 1, j - 1] = abs(horizontalGrad)

            verticalGrad = (vertical[0, 0] * img[i - 1, j - 1]) + \
                        (vertical[0, 1] * img[i - 1, j]) + \
                        (vertical[0, 2] * img[i - 1, j + 1]) + \
                        (vertical[1, 0] * img[i, j - 1]) + \
                        (vertical[1, 1] * img[i, j]) + \
                        (vertical[1, 2] * img[i, j + 1]) + \
                        (vertical[2, 0] * img[i + 1, j - 1]) + \
                        (vertical[2, 1] * img[i + 1, j]) + \
                        (vertical[2, 2] * img[i + 1, j + 1])

            newverticalImage[i - 1, j - 1] = abs(verticalGrad)

            mag = np.sqrt(pow(horizontalGrad, 2.0) + pow(verticalGrad, 2.0))
            newgradientImage[i - 1, j - 1] = mag

    cv2.imwrite("/home/yassg4mer/Project/tp_tmn/partie2/sobeleX.jpg", newhorizontalImage)
    cv2.imwrite("/home/yassg4mer/Project/tp_tmn/partie2/sobeleY.jpg", newverticalImage)
    cv2.imwrite("/home/yassg4mer/Project/tp_tmn/partie2/sobele.jpg", newgradientImage)

def SeuilSim(img1, img2):
    height, width = img.shape[:2]
    resultat_image = copy.copy(img)
    saul=float(input('seuil (float number): '))

    for i in range(height):
        for j in range(width):
            grad = np.sqrt(pow(img1[i, j], 2.0) + pow(img2[i, j], 2.0))
            
            if (grad[0] >= saul):
                resultat_image[i, j] = 1
            else:
                resultat_image[i, j] = 0

    bw = cv2.threshold(resultat_image, 0, 255, cv2.THRESH_BINARY)[1]
    cv2.imwrite("/home/yassg4mer/Project/tp_tmn/partie2/sauillage simple.jpg", bw)

def SeuilHys(img1, img2):
    Ps = float(input('seuil haut : '))
    Ph = float(input('seuil bas : '))
    height, width = img.shape[:2]
    resultat_image_ph = copy.copy(img)
    resultat_image_ps = copy.copy(img)
    newgradientImage = np.zeros((height, width))
    for i in range(height):
        for j in range(width):
            grad = np.sqrt(pow(img1[i, j], 2.0) + pow(img2[i, j], 2.0))
            if (grad[0] >= Ph):
                resultat_image_ph[i, j] = 1
            else:
                resultat_image_ph[i, j] = 0
            

    
            if (grad[0] >= Ps):
                resultat_image_ps[i, j] = 1
            else:
                resultat_image_ps[i, j] = 0

            mag = np.sqrt(pow(resultat_image_ph[i, j], 2.0) + pow(resultat_image_ps[i, j], 2.0))
            newgradientImage[i - 1, j - 1] = mag


    bw1 = cv2.threshold(resultat_image_ps, 0, 255, cv2.THRESH_BINARY)[1]
    bw2 = cv2.threshold(resultat_image_ph, 0, 255, cv2.THRESH_BINARY)[1]

    


    cv2.imwrite("/home/yassg4mer/Project/tp_tmn/partie2/sauillage ph.jpg", bw2)
    cv2.imwrite("/home/yassg4mer/Project/tp_tmn/partie2/sauillage ps.jpg", bw1)
    cv2.imwrite("/home/yassg4mer/Project/tp_tmn/partie2/sauillagephs.jpg", newgradientImage)

def LOG(img):
    height, width = img.shape[:2]
    sigma = float(input('sigma : '))
    img = cv2.GaussianBlur(img, (3, 3), sigma)
    laplace = np.array([[-1., -1., -1.],
                        [-1., 8., -1.],
                        [-1., -1., -1.]])

    newImage = np.zeros((height, width))
    for i in range(1, height - 1):
        for j in range(1, width - 1):
            imgPlacien = (laplace[0, 0] * img[i - 1, j - 1]) + \
                            (laplace[0, 1] * img[i - 1, j]) + \
                            (laplace[0, 2] * img[i - 1, j + 1]) + \
                            (laplace[1, 0] * img[i, j - 1]) + \
                            (laplace[1, 1] * img[i, j]) + \
                            (laplace[1, 2] * img[i, j + 1]) + \
                            (laplace[2, 0] * img[i + 1, j - 1]) + \
                            (laplace[2, 1] * img[i + 1, j]) + \
                            (laplace[2, 2] * img[i + 1, j + 1])

            newImage[i - 1, j - 1] = abs(imgPlacien)

    cv2.imwrite("/home/yassg4mer/Project/tp_tmn/partie2/log.jpg", newImage)

if __name__ == "__main__":
    while True:
        title = 'Please choose your favorite image processing: '
        options = ['Bruit Poiver et sell', 'Bruit Gaussien', 'Continue >']
        option, index = pick(options, title, indicator='=>', default_index=2)
        match index:
            case 0:
               BruiteProiveEtSel()
            case 1:
                BruitGaussine()
            case 2:
                break
    
    while True:
        title = 'Please choose your favorite image processing: '
        options = ['Sobel', 'Robert', 'Prewitt', 'Gradient', 'Continue >', 'Exit']
        option, index = pick(options, title, indicator='=>', default_index=2)
        match index:
            case 0:
                sobel(img)
            case 1:
                roberts(img)
            case 2:
                prewitt(img)
            case 3:
                gradient(img)
            case 4:
                break
            case 5:
                exit()    

    while True:
        title = 'Please choose your favorite image processing: '
        options = ['Sauillage Simple', 'Sauillage Hystérésis', 'LOG', 'Exit']
        option, index = pick(options, title, indicator='=>', default_index=2)
        imgPX = cv2.imread('/home/yassg4mer/Project/tp_tmn/partie2/PrewittX.jpg')
        imgPY = cv2.imread('/home/yassg4mer/Project/tp_tmn/partie2/PrewittY.jpg')
        match index:
            case 0:
               SeuilSim(imgPX, imgPY)
            case 1:
                SeuilHys(imgPX, imgPY)
            case 2:
                LOG(img)
            case 3:
                break
            case 4:
                exit()