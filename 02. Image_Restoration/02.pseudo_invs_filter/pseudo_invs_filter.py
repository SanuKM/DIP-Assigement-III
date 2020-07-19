import numpy as np
import cv2
from matplotlib import pyplot as plt

# read the input image and creating the blur image using Gaussian_blur_5x5 (matrix)
in_img = plt.imread('fruits.jpg')
#kernel_Gaussian_blur_5x5 (matrix)
kernel_gauss = np.array([[1/256, 4/256, 6/256, 4/256, 1/256], 
                             [4/256, 16/256, 25/256, 16/256, 4/256], 
                             [6/256,24/256, 36/256, 24/256, 6/256], 
                             [4/256, 16/256, 24/256, 16/256, 4/256], 
                             [1/256,4/256,6/256,4/256,1/256]])

bur_img=cv2.filter2D(in_img,-1,kernel_gauss)
cv2.imwrite('blur_img.png',bur_img)

kernel_filename = 'Guss_blur_krnl_ii.png'
h = cv2.imread(kernel_filename,0)

image_filename = 'blur_img.png'
img_bgr = cv2.imread(image_filename,1)
restored = np.zeros(img_bgr.shape)

print(image_filename)
print(kernel_filename)

r=30
K=4000
Y = 90
p = kernel_gauss

for i in range (0,3):
            g = img_bgr[:,:,i]
            G =  (np.fft.fft2(g))
            
            h_padded = np.zeros(g.shape) 
            h_padded[:h.shape[0],:h.shape[1]] = np.copy(h)
            H =  (np.fft.fft2(h_padded))
            
            p_padded = np.zeros(g.shape) 
            p_padded[:p.shape[0],:p.shape[1]] = np.copy(p)
            P =  (np.fft.fft2(p_padded)) 

            H2 = (abs(H)**2 + Y*(abs(P)**2))/(np.conjugate(H))
            H_norm = abs(H2/H2.max())
                    
            # Inverse Filter 
            F_hat = G / H_norm
            #replace division by zero (NaN) with zeroes
            #a = np.nan_to_num(F_hat)
            f_hat = np.fft.ifft2( F_hat ) #- 50*np.ones(g.shape)
            
            restored[:,:,i] = abs(f_hat)
        
#out_filename = 'restored_out_cls.png'
#cv2.imwrite(out_filename,restored)

cv2.imwrite('restore_2.png',restored)
#cv2.imwrite(out_filename,restored)
rest = cv2.imread('restore_2.png')

# Displaying  input image, Gray Scale image, DFT of the Input Image 
images = [in_img,  bur_img, rest ]
imageTitles = ['I/P img','blur_img','restor_img']

for i in range(len(images)):
    plt.subplot(1, 3, i + 1)
    plt.imshow(images[i], cmap='gray')
    plt.title(imageTitles[i])
    plt.xticks([])
    plt.yticks([])
plt.show()