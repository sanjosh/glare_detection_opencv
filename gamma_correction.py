
import cv2 as cv
import numpy as np
import os

gamma = 50
lookuptable = np.empty((1, 256), np.uint8)
lookuptable[0, :] = [np.clip(pow(i/255.0, 10)*255.0, 0, 255) for i in range(256)]

img_file = 'chaayos_pic.jpg'

img = cv.imread(img_file, cv.IMREAD_GRAYSCALE)
res = cv.LUT(img, lookuptable)
filename_gamma = f"{img_file.split('.')[0]}_{gamma}.jpg"
cv.imwrite(filename_gamma, res)

__, ret = cv.threshold(res, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)
filename_bin = f"{img_file.split('.')[0]}_{gamma}_bin.jpg"
cv.imwrite(filename_bin, ret)
cv.imshow("file", ret)
cv.waitKey()


"""
https://stackoverflow.com/questions/68610607/how-to-detect-a-flash-glare-in-an-image-of-document-using-skimage-opencv-in
"""
