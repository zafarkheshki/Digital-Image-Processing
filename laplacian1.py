#Question - 3
#Laplacian filter
import numpy as np
import cv2

img = cv2.imread('images/lena.bmp', cv2.IMREAD_GRAYSCALE)

kernel_lap = np.matrix('0 -1 0; 1 -4 1; 0 1 0')*(0.25)
out_lap = cv2.filter2D(img, -1, kernel_lap)
out_enh = img.copy()-out_lap.copy()
out_enh = out_enh.astype(np.uint8)

cv2.namedWindow('Laplacian', cv2.WINDOW_NORMAL)
cv2.imshow('Laplacian',out_lap)

cv2.namedWindow('Enhanced', cv2.WINDOW_NORMAL)
cv2.imshow('Enhanced',out_enh)

cv2.waitKey(0)
cv2.destroyAllWindows()


