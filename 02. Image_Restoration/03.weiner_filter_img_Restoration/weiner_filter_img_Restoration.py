import numpy as np
import cv2
from matplotlib import pyplot as plt

# read the input image and creating the blur image using Gaussian_blur_5x5 (matrix)
in_img = plt.imread('lena512color.tiff')
kernel_gauss = np.array([[1/256, 4/256, 6/256, 4/256, 1/256], 
                             [4/256, 16/256, 25/256, 16/256, 4/256], 
                             [6/256,24/256, 36/256, 24/256, 6/256], 
                             [4/256, 16/256, 24/256, 16/256, 4/256], 
                             [1/256,4/256,6/256,4/256,1/256]])

bur_img=cv2.filter2D(in_img,-1,kernel_gauss)
cv2.imwrite('blur_img.png',bur_img)

#constant estimate
K = 35000*25

for kernels in range (1,5):
    
    kernel_filename = 'Guss_blur_krnl_ii.png'
    h = cv2.imread(kernel_filename,0)
    #h = plt.imread(kernel_filename)
    for images in range (1,5):
        #image_filename = 'girl.png'
        image_filename = 'blur_img.png'
        img_bgr = cv2.imread(image_filename,1)
        restored = np.zeros(img_bgr.shape)
        
        #for each channel (R,G,B)
        for i in range (0,3):
            #read image and compute FFT
            g = img_bgr[:,:,i]
            G =  (np.fft.fft2(g))
            
            #2. pad kernels with zeros and compute fft
            h = cv2.imread(kernel_filename,0)
            
            h_padded = np.zeros(g.shape) 
            h_padded[:h.shape[0],:h.shape[1]] = np.copy(h)
            H =  (np.fft.fft2(h_padded))
            # normalize to [0,1]
            
            #3. Find the inverse filter term
            weiner_term = (abs(H)**2 + K)/(abs(H)**2)
            print("max value of abs(H)**2 is ",(abs(H)**2).max())
            H_weiner = H*weiner_term
            # normalize to [0,1]
            H_norm = H_weiner/abs(H_weiner.max())
            
            G_norm = G/abs(G.max())
            F_temp = G_norm/H_norm
            F_norm = F_temp/abs(F_temp.max())
            
            #rescale to original scale
            F_hat  = F_norm*abs(G.max())
            
            f_hat = np.fft.ifft2( F_hat )
            restored[:,:,i] = abs(f_hat)

        #out_filename = 'restore_1.png'
        cv2.imwrite('restore_3.png',restored)
        #matplotlib.image.imsave('restore_3.png',restored)
    rest = cv2.imread('restore_3.png')

# Displaying  OutPUT
images = [in_img,  bur_img, rest ]
imageTitles = ['I/P img','blur_img','restor_img']


for i in range(len(images)):
    plt.subplot(1, 3, i + 1)
    plt.imshow(images[i], cmap='gray')
    plt.title(imageTitles[i])
    plt.xticks([])
    plt.yticks([])
plt.show()
