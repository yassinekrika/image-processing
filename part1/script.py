import random
import cv2
import math
from PIL import Image, ImageTk
# import tkinter as tk 
import numpy as np
0

# poivre et sel
def poivre_sel(img, p):
    m , n = img.shape
    number_of_pixels = math.ceil(m*n*p)
    for i in range(number_of_pixels):
        y=random.randint(0, m - 1)
        x=random.randint(0, m - 1)
        img[y][x] = 255
         
    number_of_pixels = random.randint(300 , 10000)
    for i in range(number_of_pixels):
        y=random.randint(0, m - 1)
        x=random.randint(0, m - 1)
        img[y][x] = 0
         
    return img
 
# bruit gaussien 
def bruit_gaussien(img, sigma):

    def calcule_bruit_gaussien(sigma):
        u1 = random.random()
        u2 = random.random()
        resultat = sigma * math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)
        return resultat

    m, n = img.shape[:2]

    for i in range(m - 1):
        for j in range(n - 1):
            v = int(math.floor(img[i, j] + calcule_bruit_gaussien(sigma)))

            if (v > 255):
                v = 255
            if (v < 0):
                v = 0
            img[i, j] = v

    return img

# filtre moyen
def filtre_moyen(img, t):
    m, n = img.shape
    mask = np.ones([t, t], dtype = int)
    mask = mask / t**2

    img_new = np.zeros([m, n])
    for i in range(1, m-1):
        for j in range(1, n-1):
            temp = img[i-1, j-1]*mask[0, 0]+img[i-1, j]*mask[0, 1]+img[i-1, j + 1]*mask[0, 2]+img[i, j-1]*mask[1, 0]+ img[i, j]*mask[1, 1]+img[i, j + 1]*mask[1, 2]+img[i + 1, j-1]*mask[2, 0]+img[i + 1, j]*mask[2, 1]+img[i + 1, j + 1]*mask[2, 2]
            img_new[i, j]= temp

    return img_new

# filtre median
def filtre_median(img, t):
    m, n = img.shape
    img_new = np.zeros([m, n])
    mask = np.zeros((t, t))
    for i in range(1, m-1):
        for j in range(1, n-1):
            temp = [img[i-1, j-1],
                img[i-1, j],
                img[i-1, j + 1],
                img[i, j-1],
                img[i, j],
                img[i, j + 1],
                img[i + 1, j-1],
                img[i + 1, j],
                img[i + 1, j + 1]]
            
            temp1 = sorted(temp)
            img_new[i, j]= temp1[4]

    return img_new

# PSNR
def peack_signal_noise_ration(img_origin, img_bruit):
    m, n = img_origin.shape
    some = 0
    r = 255
    for i in range(1, m-1):
        for j in range(1, n-1):
            some = some + (img_origin[i, j] - img_bruit[i, j]) ** 2

    pnsr = 10 * math.log10(r ** 2 / (some / (m * n)))
    print('peack signal noise ration (PSNR) :', pnsr)


#Storing the image
img = cv2.imread('/home/yassg4mer/Project/tp_tmn/partie2/m.jpg', cv2.IMREAD_GRAYSCALE)
si = float(input('add sigma value : '))
pi = float(input('add p value : '))
cv2.imwrite('salt-and-pepper-lena.jpg', poivre_sel(img, pi))
cv2.imwrite('bruit_gaussien.jpg', bruit_gaussien(img, si))


x1 = Image.open('m.jpg')
x2 = Image.open('bruit_gaussien.jpg')
x3 = Image.open('salt-and-pepper-lena.jpg')

# recall the modified images x1 & x2
img_x1 = cv2.imread('bruit_gaussien.jpg', cv2.IMREAD_GRAYSCALE)
img_x2 = cv2.imread('salt-and-pepper-lena.jpg', cv2.IMREAD_GRAYSCALE)
cv2.imwrite('filtre_moyen.jpg', filtre_moyen(img_x1, 7))
cv2.imwrite('filtre_median.jpg', filtre_median(img_x2, 3))

y2 = Image.open('filtre_moyen.jpg')
y3 = Image.open('filtre_median.jpg')

x1.show()
x2.show()
x3.show()
y2.show()
y3.show()

img_y3 = cv2.imread('filtre_median.jpg', cv2.IMREAD_GRAYSCALE)
peack_signal_noise_ration(img, img_y3)



# build the ui 
