import cv2
from matplotlib import pyplot as plt

# Load the image
image = cv2.imread("lena512.bmp")
# Blur the image
gauss = cv2.GaussianBlur(image, (7,7), 0)
# Apply Unsharp masking
unsharp_image = cv2.addWeighted(image, 2, gauss, -1, 0)

# Displaying  input image, Gray Scale image, DFT of the Input Image 
images = [image,unsharp_image ]
imageTitles = ['input image','unsharp_masking_image image' ]

for i in range(len(images)):
    plt.subplot(1, 2, i + 1)
    plt.imshow(images[i], cmap='gray')
    plt.title(imageTitles[i])
    plt.xticks([])
    plt.yticks([])
plt.show()
