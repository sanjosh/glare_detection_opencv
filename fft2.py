

import cv2
import numpy as np
from matplotlib import pyplot as plt


# def do_notch(IMs):
#     N = IMs.shape[0]
#     print(IMs.shape)
#     x, y = np.meshgrid(np.arange(N), np.arange(N))
#
#     # notch filter generation (need to understand)
#
#     a1 = 0.008
#     a2 = 0.008
#
#     NF1 = 1 - np.exp(-a1 * (x - 190) ** 2 - a2 * (y - 123) ** 2)  # Gaussian
#     NF2 = 1 - np.exp(-a1 * (x - 104) ** 2 - a2 * (y - 172) ** 2)  # Gaussian
#     NF3 = 1 - np.exp(-a1 * (x - 126) ** 2 - a2 * (y - 135) ** 2)  # Gaussian
#     NF4 = 1 - np.exp(-a1 * (x - 168) ** 2 - a2 * (y - 161) ** 2)  # Gaussian
#
#     Z = NF1 * NF2 * NF3 * NF4
#     IMFs = IMs * Z
#
#     IMFr = np.fft.ifftshift(IMFs)
#     imfr = np.fft.ifft2(IMFr)
#
#     plt.subplot(2, 2, 3)
#     plt.title('Filtered Image')
#     plt.imshow(np.real(imfr), cmap='gray')
#     plt.axis('off')

img = cv2.imread('Picture5.jpg',0)
f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)

# log transform
magnitude_spectrum = 20*np.log(np.abs(fshift))

# th2 = cv2.threshold(magnitude_spectrum, 127, 255, cv2.THRESH_BINARY)


plt.subplot(121), plt.imshow(img, cmap='gray')
plt.title('Input Image')
plt.xticks([]), plt.yticks([])
plt.subplot(122)
plt.imshow(magnitude_spectrum, cmap = 'gray')
plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])

# do_notch(fshift)
plt.show()


