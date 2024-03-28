import cv2
import numpy as np

img = cv2.imread('IMG_3926.jpg')
hh, ww = img.shape[:2]

# threshold
lower = (150,150,150)
upper = (240,240,240)
thresh = cv2.inRange(img, lower, upper)

# apply morphology close and open to make mask
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25,25))
morph = cv2.morphologyEx(morph, cv2.MORPH_DILATE, kernel, iterations=1)

# floodfill the outside with black
black = np.zeros([hh + 2, ww + 2], np.uint8)
mask = morph.copy()
mask = cv2.floodFill(mask, black, (0,0), 0, 0, 0, flags=8)[1]

cv2.imshow("IMAGE", img)
cv2.imshow("THRESH", thresh)
cv2.imshow("MORPH", morph)
cv2.imshow("MASK", mask)
# cv2.imshow("RESULT1", result1)
# cv2.imshow("RESULT2", result2)
cv2.waitKey(0)

"""
https://stackoverflow.com/questions/72514384/how-can-i-remove-the-bright-glare-regions-in-image
"""