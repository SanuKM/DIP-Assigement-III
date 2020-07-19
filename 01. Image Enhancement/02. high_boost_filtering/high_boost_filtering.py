# program to enhance the image using  High-boost filtering.

import cv2
import numpy as np
from matplotlib import pyplot as plt

image = cv2.imread("eye.jpg", cv2.COLOR_BGR2GRAY)
cv2.imshow("Input", image)

kernel_i = np.array([[-1, -1, -1],
                   [-1,  8, -1],
                   [-1, -1, -1]])
#High Pass Kernel 5x5
kernel_ii = np.array([[-1, -1, -1, -1, -1],
                   [-1,  1,  2,  1, -1],
                   [-1,  2,  4,  2, -1],
                   [-1,  1,  2,  1, -1],
                   [-1, -1, -1, -1, -1]])

# Note
# High Pass Filters can also be obtained by subtracting a low pass filtered image
# from original image
# Using a LPF like Gaussian Filter
# image_hpf = image - gauss_mask

image_hpf_i = cv2.filter2D(image, -1, kernel_i)
image_hpf_ii = cv2.filter2D(image, -1, kernel_ii)

cv2.imshow("Output : High Pass Filter 3x3", image_hpf_i)
cv2.imshow("Output : High Pass Filter 5x5", image_hpf_ii)


cv2.imwrite("/home/sanu/Admin/M.Tech/Lab/Sanu Python Prgm Cycle III/02. high_boost_filtering/Output:HPF_3x3.png",image_hpf_i)
cv2.imwrite("/home/sanu/Admin/M.Tech/Lab/Sanu Python Prgm Cycle III/02. high_boost_filtering/Output_5x5.png",image_hpf_ii)
cv2.imwrite("/home/sanu/Admin/M.Tech/Lab/Sanu Python Prgm Cycle III/02. high_boost_filtering/Input:eye.png",image)
cv2.waitKey(0)