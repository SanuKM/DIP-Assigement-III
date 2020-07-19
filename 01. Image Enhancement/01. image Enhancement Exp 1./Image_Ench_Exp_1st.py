import numpy as np
import cv2
from matplotlib import pyplot as plt
from skimage.util import random_noise

 #Add salt-and-pepper noise to the image.
def salt_pepper(imgs):
    noise_img = random_noise(imgs, mode='s&p',amount=0.3)
    # The above function returns a floating-point image
    # on the range [0, 1], thus we changed it to 'uint8' and from [0,255]
    noise = np.array(255*noise_img, dtype = 'uint8')
    return (noise)

#ApplyAveraging filtering for remove noise
def avg_fltr_3x3(imgs):
    noise = imgs
    
    #kernel = np.ones((3,3),dtype=np.float32)/9   
    a1, a2 = noise.shape
    mask = np.ones((3,3),dtype=np.float32)/9

    avg_img = np.zeros([a1,a2])
    for i in range(1, a1-1): 
        for j in range(1, a2-1): 
            temp = noise[i-1, j-1]*mask[0, 0]+noise[i-1, j]*mask[0, 1]+noise[i-1, j + 1]*mask[0, 2]+noise[i, j-1]*mask[1, 0]+ noise[i, j]*mask[1, 1]+noise[i, j + 1]*mask[1, 2]+noise[i + 1, j-1]*mask[2, 0]+noise[i + 1, j]*mask[2, 1]+noise[i + 1, j + 1]*mask[2, 2] 
            avg_img[i,j] = temp

    AvgFilter = avg_img.astype(np.uint8)  
    #AvgFilterk = cv2.filter2D(noise,-1,kernel)
    #cv2.imshow('AvgFilterk3x3',AvgFilterk)
    return (AvgFilter)

def avg_fltr_5x5(imgs):
    noise = imgs    
    #kernel = np.ones((5,5),dtype=np.float32)/9
    
    a1, a2 = noise.shape
    mask = np.ones((5,5),dtype=np.float32)/9

    avg_img = np.zeros([a1,a2])
    for i in range(1, a1-1): 
        for j in range(1, a2-1): 
            temp = noise[i-1, j-1]*mask[0, 0]+noise[i-1, j]*mask[0, 1]+noise[i-1, j + 1]*mask[0, 2]+noise[i, j-1]*mask[1, 0]+ noise[i, j]*mask[1, 1]+noise[i, j + 1]*mask[1, 2]+noise[i + 1, j-1]*mask[2, 0]+noise[i + 1, j]*mask[2, 1]+noise[i + 1, j + 1]*mask[2, 2] 
            avg_img[i,j] = temp

    AvgFilter = avg_img.astype(np.uint8)  
    #AvgFilterk = cv2.filter2D(noise,-1,kernel)
    #cv2.imshow('AvgFilterk5x5',AvgFilterk)
    return (AvgFilter)

#Apply weighted Averaging filtering for remove noise
def weighted_avg_fltr_3x3(imgs):
    noise = imgs
    kernel = np.ones((3,3),dtype=np.float32)
    kernel[2, 2] = 10.0
    kernel /= 34
    WeightedAvgFilter = cv2.filter2D(noise,-1,kernel)
    return (WeightedAvgFilter)

def weighted_avg_fltr_5x5(imgs):
    noise = imgs
    kernel = np.ones((5,5),dtype=np.float32)
    kernel[2, 2] = 10.0
    kernel /= 34
    WeightedAvgFilter = cv2.filter2D(noise,-1,kernel)
    return (WeightedAvgFilter)

#Apply Gaussian filter
def GaussianBlur_3x3(imgs):
    noise = imgs   
    gauss_img = cv2.GaussianBlur(noise,(3,3),cv2.BORDER_DEFAULT)
    return (gauss_img)

def GaussianBlur_5x5(imgs):
    noise = imgs   
    gauss_img = cv2.GaussianBlur(noise,(5,5),cv2.BORDER_DEFAULT)
    return (gauss_img)

#Apply Median filter
def median_fltr_3x3(imgs):
    noise = imgs
    median_img = cv2.medianBlur(noise, 3)
    return (median_img)

def median_fltr_5x5(imgs):
    noise = imgs
    median_img = cv2.medianBlur(noise, 5)
    return (median_img)


if __name__ == '__main__': 
    input_img = 'boat.512.tiff'

    in_img = cv2.imread(input_img,0)
    noise_img = salt_pepper(in_img)
    avg_img3x3 = avg_fltr_3x3(in_img)
    avg_img5x5 = avg_fltr_5x5(in_img)
    weg_img3x3 = weighted_avg_fltr_3x3(in_img)
    weg_img5x5 = weighted_avg_fltr_5x5(in_img)
    gaus_img3x3 = GaussianBlur_3x3(in_img)
    gaus_img5x5 = GaussianBlur_5x5(in_img)
    med_img3x3 = median_fltr_3x3(in_img)
    med_img5x5 = median_fltr_5x5(in_img)


    cv2.imshow('input_img',in_img)
    cv2.imshow('noise_img',noise_img)
    cv2.imshow('avg_img3x3',avg_img3x3)
    cv2.imshow('avg_img5x5',avg_img5x5)
    cv2.imshow('weg_img3x3',weg_img3x3)
    cv2.imshow('weg_img5x5',weg_img5x5)
    cv2.imshow('gaus_img3x3',gaus_img3x3)
    cv2.imshow('gaus_img5x5',gaus_img5x5)
    cv2.imshow('med_img3x3',med_img3x3)
    cv2.imshow('med_img5x5',med_img5x5)


    #cv2.imwrite("/home/sanu/Admin/M.Tech/Lab/Sanu Python Prgm Cycle III/p/input_img.png",in_img)
    #cv2.imwrite("/home/sanu/Admin/M.Tech/Lab/Sanu Python Prgm Cycle III/p/noise_img.png",noise_img)
    #cv2.imwrite("/home/sanu/Admin/M.Tech/Lab/Sanu Python Prgm Cycle III/p/avg_img3x3.png",avg_img3x3)
    #cv2.imwrite("/home/sanu/Admin/M.Tech/Lab/Sanu Python Prgm Cycle III/p/avg_img5x5.png",avg_img5x5)
    #cv2.imwrite("/home/sanu/Admin/M.Tech/Lab/Sanu Python Prgm Cycle III/p/weg_img3x3.png",weg_img5x5)
    #cv2.imwrite("/home/sanu/Admin/M.Tech/Lab/Sanu Python Prgm Cycle III/p/weg_img5x5.png",weg_img5x5)
    #cv2.imwrite("/home/sanu/Admin/M.Tech/Lab/Sanu Python Prgm Cycle III/p/gaus_img3x3.png",gaus_img3x3)
    #cv2.imwrite("/home/sanu/Admin/M.Tech/Lab/Sanu Python Prgm Cycle III/p/gaus_img5x5.png",gaus_img5x5)
    #cv2.imwrite("/home/sanu/Admin/M.Tech/Lab/Sanu Python Prgm Cycle III/p/med_img3x3.png",med_img3x3)
    #cv2.imwrite("/home/sanu/Admin/M.Tech/Lab/Sanu Python Prgm Cycle III/p/med_img5x5.png",med_img5x5)      

    cv2.waitKey(0)

    cv2.destroyAllWindows()



 

   
