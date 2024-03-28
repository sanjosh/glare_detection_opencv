
import cv2
lum = cv2.imread('glare.jpeg',cv2.IMREAD_GRAYSCALE)

cv2.imshow("LUM", lum)
cv2.waitKey()

"""
https://stackoverflow.com/questions/59280375/how-to-get-luminance-gradient-of-an-image
"""
