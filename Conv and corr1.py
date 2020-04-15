#Question - 2
#Correlation and Convolution filters
import numpy as np
import cv2

img = cv2.imread('images/lena.bmp', cv2.IMREAD_GRAYSCALE)

kernel_corr = np.matrix('1 -1 -1; 1 2 -1; 1 1 1')*(0.25)
kernel_conv = kernel_corr.transpose()

out_corr = cv2.filter2D(img, -1, kernel_corr)
out_conv = cv2.filter2D(img, -1, kernel_conv)

cv2.namedWindow('Correlation', cv2.WINDOW_NORMAL)
cv2.imshow('Correlation',out_corr)

cv2.namedWindow('Convolution', cv2.WINDOW_NORMAL)
cv2.imshow('Convolution',out_conv)

cv2.waitKey(0)
cv2.destroyAllWindows()


