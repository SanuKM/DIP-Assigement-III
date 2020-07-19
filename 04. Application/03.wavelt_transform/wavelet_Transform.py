import numpy as np
import matplotlib.pyplot as plt
import cv2
import pywt
import pywt.data

# Load image

original = cv2.imread('lena512color.tiff',0)

# Wavelet transform of image, and plot approximation and details
titles = ['input_img','Approximation', ' Horizontal detail',
          'Vertical detail', 'Diagonal detail']
coeffs2 = pywt.dwt2(original, 'bior1.3')
LL, (LH, HL, HH) = coeffs2
fig = plt.figure(figsize=(12, 3))


for i, a in enumerate([original,LL, LH, HL, HH]):
    ax = fig.add_subplot(1, 5, i + 1)
    ax.imshow(a, interpolation="nearest", cmap=plt.cm.gray)
    ax.set_title(titles[i], fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])

fig.tight_layout()
plt.show()
