import numpy as np
import cv2

#Reads image from file
img = cv2.imread('images/lena.bmp')
gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

filter_order = 1

kernel_size = 5
number_of_elements = kernel_size*kernel_size

kernel_mean = np.ones((kernel_size,kernel_size),np.float32)/number_of_elements



out_mean = gray_image.copy()
out_median = gray_image.copy()

for x in range(0, filter_order):
    out_mean = cv2.filter2D(out_mean,-1,kernel_mean)
    out_median = cv2.medianBlur(out_median,kernel_size)
    

#Displys Image
cv2.namedWindow('Mean', cv2.WINDOW_NORMAL)
cv2.imshow('Mean',out_mean)
cv2.namedWindow('Median', cv2.WINDOW_NORMAL)
cv2.imshow('Median',out_median)
cv2.waitKey(0)              #watis for any keyboard event

