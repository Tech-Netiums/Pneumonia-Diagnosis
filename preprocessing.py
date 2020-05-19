from PIL import Image, ImageChops
from scipy import ndimage
import numpy as np


"""img_file_test = '/Users/julianakani-guery/Desktop/datasets_17810_23812_chest_xray_train_NORMAL_IM-0115-0001.jpeg'
img_test = Image.open(img_file_test)"""
passe_haut = [[0,-4,0],[-4,18,-4],[0,-4,-0]]
passe_bas = [[1,1,1],[1,6,1],[1,1,1]]
laplacian_of_gaussian_33 =  [[0,-1,0],[-1,4,1],[0,-1,0]] #interessant, à tester
laplacian_of_gaussian_55 = [[0,0,-1,0,0],[0,-1,-2,-1,0],[-1,-2,16,-2,-1],[0,-1,-2,-1,0],[0,0,-1,0,0]] #rendu naze


def invert(img) : #renvoit le négatif d'img
    return ImageChops.invert(img)


def convolution2D(img, kernel):
    img_array = np.array(img)
    conv_array = ndimage.convolve(img_array, np.asarray(kernel), mode='constant', cval=0.0)
    img_conv = Image.fromarray(conv_array)
    return (img_conv)
