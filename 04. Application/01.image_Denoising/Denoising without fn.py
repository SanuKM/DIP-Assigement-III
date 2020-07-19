import numpy as np
from matplotlib import pyplot as plt
import cv2

# Original Image
lenna_img = cv2.imread('Lenna.png', cv2.IMREAD_GRAYSCALE)
#original image : converting it to binary values (0 or 1)
lenna_img_bw = np.zeros(lenna_img.shape)
lenna_img_bw[lenna_img > 128] = 1
cv2.imshow('input img',lenna_img_bw)

# Noisy Image We can now produce a noisy image by flipping each pixel with a probability of 0.3.

p = 0.3
lenna_noisy = np.zeros(lenna_img_bw.shape)
for i in range(512):
    for j in range(512):
        if np.random.binomial(1, p): # Flip value
            if lenna_img_bw[i, j]:
                lenna_noisy[i, j] = 0
            else:
                lenna_noisy[i, j] = 1
        else:
            lenna_noisy[i, j] = lenna_img_bw[i, j]

cv2.imshow('noisy_img',lenna_noisy)

#Recover Original Image  x, y are the are initialised and the paramters, h, beta, eta are the set to appropriate values,
# Finally, we run over the set of pixels 10 times and evaluate which binary value achieves a
y = lenna_noisy
x = y # Initialisation of x

x[x==0] = -1
y[y==0] = -1

# Set parameters
h = 0
beta = 2.0
eta = 0.5

def local_energy(i, j, x, y):
    energy = h * x[i, j]
    s = 0
    for k in [(i-1, j), (i+1, j), (i, j-1), (i, j+1)]:
        if k[0] >= 0 and k[0] <= 511 and k[1] >= 0 and k[1] <= 511:
            s += x[k]
    energy -= eta * x[i, j] * s
    energy -= beta * x[i, j] * y[i, j]
    return energy

for k in range(10):
    for i in range(512):
        for j in range(512):
            le1 = local_energy(i, j, x, y)
            x[i, j] = -x[i, j]
            le2 = local_energy(i, j, x, y)
            if le1 < le2:
                x[i, j] = -x[i, j]
    print('Done with epoch', k)

x[x==-1] = 0
cv2.imshow('Recovered Image',x)
cv2.waitKey(0)
cv2.destroyAllWindows()

