import cv2
import numpy as np

img = cv2.imread('cam.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_gaussian = cv2.GaussianBlur(gray,(3,3),0)


#roberts
roberts_cross_v = np.array( [[ 0, 0, 0 ],
                             [ 0, 1, 0 ],
                             [ 0, 0,-1 ]] )

roberts_cross_h = np.array( [[ 0, 0, 0 ],
                             [ 0, 0, 1 ],
                             [ 0,-1, 0 ]] )
rob_Voutput=cv2.filter2D(img_gaussian,-1,roberts_cross_v)
rob_Houtput=cv2.filter2D(img_gaussian,-1,roberts_cross_h)
img_robert = rob_Voutput + rob_Houtput

#canny
img_canny = cv2.Canny(img,100,200)

#sobel
#roberts_cross
sobel_kern_y = np.array( [[ -1, 0, 1 ],
                             [ -2, 1, 2 ],
                             [ -1, 0, 1 ]] )

sobel_kern_x = np.array( [[ 1, 2, 1 ],
                             [ 0, 0, 0 ],
                             [ -1,-2, -1 ]] )

img_sobelx = cv2.filter2D(img_gaussian,-1,sobel_kern_x)
img_sobely = cv2.filter2D(img_gaussian,-1,sobel_kern_y)
#img_sobelx = cv2.Sobel(img_gaussian,cv2.CV_8U,1,0,ksize=5)
#img_sobely = cv2.Sobel(img_gaussian,cv2.CV_8U,0,1,ksize=5)
img_sobel = img_sobelx + img_sobely


#prewitt
kernelx = np.array([[1,1,1],[0,0,0],[-1,-1,-1]])
kernely = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
img_prewittx = cv2.filter2D(img_gaussian, -1, kernelx)
img_prewitty = cv2.filter2D(img_gaussian, -1, kernely)
img_prewitt = img_prewittx + img_prewitty

cv2.imshow("Original Image", img)
cv2.imshow("roberts x",rob_Houtput)
cv2.imshow("roberts y",rob_Voutput)
cv2.imshow("roberts",img_robert)
cv2.imshow("Canny", img_canny)
cv2.imshow("Sobel X", img_sobelx)
cv2.imshow("Sobel Y", img_sobely)
cv2.imshow("Sobel", img_sobel)
cv2.imshow("Prewitt X", img_prewittx)
cv2.imshow("Prewitt Y", img_prewitty)
cv2.imshow("Prewitt", img_prewittx + img_prewitty)


cv2.imwrite("/home/sanu/Admin/M.Tech/Lab/Sanu Python Prgm Cycle III/08. Segmentation/Input_Image.png", img)
cv2.imwrite("/home/sanu/Admin/M.Tech/Lab/Sanu Python Prgm Cycle III/08. Segmentation/roberts x.png",rob_Houtput)
cv2.imwrite("/home/sanu/Admin/M.Tech/Lab/Sanu Python Prgm Cycle III/08. Segmentation/roberts y.png",rob_Voutput)
cv2.imwrite("/home/sanu/Admin/M.Tech/Lab/Sanu Python Prgm Cycle III/08. Segmentation/roberts.png",img_robert)
cv2.imwrite("/home/sanu/Admin/M.Tech/Lab/Sanu Python Prgm Cycle III/08. Segmentation/Canny.png", img_canny)
cv2.imwrite("/home/sanu/Admin/M.Tech/Lab/Sanu Python Prgm Cycle III/08. Segmentation/Sobel X.png", img_sobelx)
cv2.imwrite("/home/sanu/Admin/M.Tech/Lab/Sanu Python Prgm Cycle III/08. Segmentation/Sobel Y.png", img_sobely)
cv2.imwrite("/home/sanu/Admin/M.Tech/Lab/Sanu Python Prgm Cycle III/08. Segmentation/Sobel.png", img_sobel)
cv2.imwrite("/home/sanu/Admin/M.Tech/Lab/Sanu Python Prgm Cycle III/08. Segmentation/Prewitt X.png", img_prewittx)
cv2.imwrite("/home/sanu/Admin/M.Tech/Lab/Sanu Python Prgm Cycle III/08. Segmentation/Prewitt Y.png", img_prewitty)
cv2.imwrite("/home/sanu/Admin/M.Tech/Lab/Sanu Python Prgm Cycle III/08. Segmentation/Prewitt.png", img_prewitt)


cv2.waitKey(0)
cv2.destroyAllWindows()