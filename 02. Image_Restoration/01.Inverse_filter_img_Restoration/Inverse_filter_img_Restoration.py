import numpy as np
import cv2
from matplotlib import pyplot as plt

# read the input image and creating the blur image using Gaussian_blur_5x5 (matrix)
in_img = plt.imread('dog.jpg')
#kernel_Gaussian_blur_5x5 (matrix)
kernel_gauss = np.array([[1/256, 4/256, 6/256, 4/256, 1/256], 
                             [4/256, 16/256, 25/256, 16/256, 4/256], 
                             [6/256,24/256, 36/256, 24/256, 6/256], 
                             [4/256, 16/256, 24/256, 16/256, 4/256], 
                             [1/256,4/256,6/256,4/256,1/256]])

bur_img=cv2.filter2D(in_img,-1,kernel_gauss)
cv2.imwrite('blur_img.png',bur_img)

for kernels in range (1,5):
    kernel_filename = 'Guss_blur_krnl_ii.png'
    h = cv2.imread(kernel_filename,0)
    for images in range (1,5):
        image_filename = 'blur_img.png'
        img_bgr = cv2.imread(image_filename,1)
        restored = np.zeros(img_bgr.shape)
        
        print(image_filename)
        print(kernel_filename)
        
        #for each channel (R,G,B)
        for i in range (0,3):
            #1.read image and compute fft
            g = img_bgr[:,:,i]
            G =  (np.fft.fft2(g))
            
            #2. pad kernels with zeros and compute fft
            h_padded = np.zeros(g.shape) 
            h_padded[:h.shape[0],:h.shape[1]] = np.copy(h)
            H =  (np.fft.fft2(h_padded))
            
            # normalize to [0,1]
            H_norm = H/abs(H.max())
            G_norm = G/abs(G.max())
            F_temp = G_norm/H_norm
            F_norm = F_temp/abs(F_temp.max())
            
            #rescale to original scale
            F_hat  = F_norm*abs(G.max())
            
            # 3. apply Inverse Filter and compute IFFT  F_hat = G / H           
            f_hat = np.fft.ifft2( F_hat )
            restored[:,:,i] = abs(f_hat)
        
        #write file         
        cv2.imwrite('restore_1.png',restored)
        #cv2.imwrite(out_filename,restored)
        rest = cv2.imread('restore_1.png')

# Displaying  input image, Gray Scale image, DFT of the Input Image 
images = [in_img,  bur_img, rest ]
imageTitles = ['I/P img','blur_img','restor_img']

for i in range(len(images)):
    plt.subplot(1, 3, i + 1)
    plt.imshow(images[i])#, cmap='gray')
    plt.title(imageTitles[i])
    plt.xticks([])
    plt.yticks([])
plt.show()