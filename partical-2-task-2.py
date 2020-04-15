import numpy as np
import cv2

#Reads image from file
img = cv2.imread('images/lena.bmp')
gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

kernel_mean = np.matrix('1 1 1; 1 1 1; 1 1 1')/9
kernel_corr = np.matrix('1 -1 -1; 1 2 -1; 1 1 1')*(0.25)
kernel_conv = kernel_corr.transpose()
kernel_lap = np.matrix('0 1 0; 1 -4 1; 0 1 0')*(0.25)

filter_order = 2

out_mean = gray_image.copy()
out_median = gray_image.copy()
out_corr = gray_image.copy()
out_conv = gray_image.copy()
out_lap = gray_image.copy()
out_enh = gray_image.copy()

for x in range(0, filter_order):
    out_mean = cv2.filter2D(out_mean,-1,kernel_mean)
    out_median = cv2.medianBlur(out_median,3)
    out_corr = cv2.filter2D(out_corr,-1,kernel_corr)
    out_conv = cv2.filter2D(out_conv,-1,kernel_conv)
    out_lap = cv2.filter2D(out_lap,-1,kernel_lap)
    out_enh = gray_image.copy()-out_lap.copy()
    out_enh = out_enh.astype(np.uint8)
    

#Displys the Images
cv2.namedWindow('Mean', cv2.WINDOW_NORMAL)
cv2.imshow('Mean',out_mean)
cv2.namedWindow('Median', cv2.WINDOW_NORMAL)
cv2.imshow('Median',out_median)
cv2.namedWindow('Correlation', cv2.WINDOW_NORMAL)
cv2.imshow('Correlation',out_corr)
cv2.namedWindow('Convolution', cv2.WINDOW_NORMAL)
cv2.imshow('Convolution',out_conv)
cv2.namedWindow('Lap', cv2.WINDOW_NORMAL)
cv2.imshow('Lap',out_lap)
cv2.namedWindow('Enhanced', cv2.WINDOW_NORMAL)
cv2.imshow('Enhanced',out_enh)
cv2.waitKey(0)              #watis for any keyboard event

