
import cv2
import numpy as np
from matplotlib import pyplot as plt

def sorter(arr, n = 10):
    xx = np.sort(arr, axis=None)

    P, Q = arr.shape
    flat_indices = np.argpartition(arr.ravel(), n - 1)
    flat_indices = flat_indices[-n:]
    row_indices, col_indices = np.unravel_index(flat_indices, arr.shape)

    min_elements = arr[row_indices, col_indices]
    min_elements_order = np.argsort(min_elements)
    row_indices, col_indices = row_indices[min_elements_order], col_indices[min_elements_order]
    row_indices = row_indices - P//2
    col_indices = col_indices - Q//2
    return row_indices, col_indices


def notch_reject_filter(shape, d0=9, u_k=0, v_k=0):
    # https://stackoverflow.com/questions/65483030/notch-reject-filtering-in-python
    (M, N) = shape

    H_0_u = np.repeat(np.arange(M), N).reshape((M, N))
    H_0_v = np.repeat(np.arange(N), M).reshape((N, M)).transpose()

    D_uv = np.sqrt((H_0_u - M / 2 + u_k) ** 2 + (H_0_v - N / 2 + v_k) ** 2)
    D_muv = np.sqrt((H_0_u - M / 2 - u_k) ** 2 + (H_0_v - N / 2 - v_k) ** 2)

    selector_1 = D_uv <= d0
    selector_2 = D_muv <= d0

    selector = np.logical_or(selector_1, selector_2)

    H = np.ones((M, N))
    H[selector] = 0

    return H


img = cv2.imread('screen_of_screen.jpg',0)
f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)

# log transform
magnitude_spectrum = 20*np.log(np.abs(fshift))

# r, c = sorter(magnitude_spectrum, 100)
img_shape = img.shape
#
# h_list = []
# for a, b in zip(r, c):
#     h_list.append(notch_reject_filter(img_shape, 4, a, b))
# #
# #
# NotchFilter = h_list[0]
# for h in h_list[1:]:
#     NotchFilter = NotchFilter * h

# dst = cv2.medianBlur(magnitude_spectrum, 7)
dst = cv2.threshold(magnitude_spectrum, 127, 255, cv2.THRESH_BINARY)


plt.subplot(131)
plt.imshow(img, cmap='gray')
plt.title('Input Image')
plt.xticks([])
plt.yticks([])

plt.subplot(132)
plt.imshow(magnitude_spectrum, cmap = 'gray')
plt.title('Magnitude Spectrum')
plt.xticks([])
plt.yticks([])

plt.subplot(133)
plt.imshow(dst, cmap = 'gray')
plt.title('other Spectrum')
plt.xticks([]), plt.yticks([])

plt.show()


