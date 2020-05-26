from PIL import Image, ImageChops
from scipy import ndimage
import numpy as np
import math

#Image test
"""img_file_test = '/Users/julianakani-guery/Desktop/PRAMA Projet/1_exemple_normal.jpeg'
img_test = Image.open(img_file_test)"""

#Filtres pour la convolution
passe_haut = [[0,-4,0],[-4,18,-4],[0,-4,0]]
passe_bas = [[1,1,1],[1,6,1],[1,1,1]]
laplacian_of_gaussian_33 =  [[0,-1,0],[-1,4,1],[0,-1,0]] #interessant, à tester
laplacian_of_gaussian_55 = [[0,0,-1,0,0],[0,-1,-2,-1,0],[-1,-2,16,-2,-1],[0,-1,-2,-1,0],[0,0,-1,0,0]] #rendu naze
sobel_vertical = [[-1,0,1],[-2,0,2],[-1,0,1]] #détection verticale des contours 
sobel_horizontal = [[-1,-2,-1],[0,0,0],[1,2,1]] #détection horizontale des contours
kirsch_vertical = [[-3,-3,5],[-3,0,5],[-3,-3,5]] #détection verticale des contours 
kirsch_horizontal = [[-3,-3,-3],[0,0,-3],[5,5,5]] #détection horizontale des contours
mdif_horizontal = [[0,-1,-1,-1,0],[-1,-2,-3,-2,-1],[0,0,0,0,0],[1,2,3,2,1],[0,1,1,1,0]] #détection horizontale des contours
mdif_vertical = [[0,-1,0,1,0],[-1,-2,0,2,1],[-1,-3,0,3,1],[-1,-2,0,2,1],[0,-1,0,1,0]] #détection verticale des contours
amelioration_nettete = [[0,-1,0],[-1,5,-1],[0,-1,0]]

def invert(img) : #renvoit le négatif d'img
    return ImageChops.invert(img)

def convolution2D(img, kernel):
    img_array = np.array(img)
    conv_array = ndimage.convolve(img_array, np.asarray(kernel), mode='constant', cval=0.0)
    #img_conv = Image.fromarray(conv_array)
    return (conv_array)

def somme_img(img1,img2):
    img1_array = np.array(img1)
    img2_array = np.array(img2)
    somme_array = img1_array + img2_array
    return(Image.fromarray(somme_array))


def median_cut(img) : #atténuation du bruit
    col,row = np.array(img).shape
    img_array = np.array(img)
    img_cut_array = img_array
    for i in range(1,col-2):
        for j in range(1,row-2):
            L=[]
            for l in [-1,0,1]:
                for m in [-1,0,1]:
                    L.append(img_array[i+l][j+m])
            L.sort()
            img_cut_array[i][j] = L[4]
    return (img_cut_array)

