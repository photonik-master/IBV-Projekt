import cv2 as cv
import matplotlib as plt
import numpy as np

def get_image(name, param):
    img = cv.imread(name, param)
    return img

if __name__ == '__main__':
    pass