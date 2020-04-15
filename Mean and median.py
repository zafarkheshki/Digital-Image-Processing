# Question - 1
# Mean and Median filters

import numpy as np
import cv2

#Read image from file
img = cv2.imread('images/lena.bmp', cv2.IMREAD_GRAYSCALE)
Mean = img.copy()
Median = img.copy()

# getting height and width of the image
width,height = img.shape[:2]

print(img[0:3,0:3])
for i in range(1,width-2):
    for j in range (1,height-2):
        Mean[i,j] = int(img[i-1:i+2,j-1:j+2].sum()/9)
        Median[i,j] = int(np.median(img[i-1:i+2,j-1:j+2]))

#Disply Mean output
cv2.namedWindow('Mean', cv2.WINDOW_NORMAL)
cv2.imshow('Mean',Mean)

#Display Median output
cv2.namedWindow('Median', cv2.WINDOW_NORMAL)
cv2.imshow('Median',Median)

cv2.waitKey(0)
cv2.destroyAllWindows()
