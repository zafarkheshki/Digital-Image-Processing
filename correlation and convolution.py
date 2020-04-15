# Question - 2
# Correlation and convolution

import numpy as np
import cv2

#Read image from file
img = cv2.imread('images/lena.bmp', cv2.IMREAD_GRAYSCALE)
Correlation = img.copy()
Convolution = img.copy()

# getting height and width of the image
width,height = img.shape[:2]

kernel = (1/4)* np.array([[1, -1, -1], [1, 2, -1], [1, 1, 1]])

print(img[0:3,0:3])
for i in range(1,width-2):
    for j in range (1,height-2):
        temp = int(np.multiply(img[i-1:i+2,j-1:j+2],kernel).sum())
        if(temp>255):
           Correlation[i,j]=255
        elif(temp<0):
           Correlation[i,j]=0
        else:
           Correlation[i,j]=temp


kernel_90 = np.rot90(kernel)
kernel_180 = np.rot90(kernel_90)
print(kernel_180)
print(img[0:3,0:3])
for i in range(1,width-2):
    for j in range (1,height-2):
        temp_2 = int(np.multiply(img[i-1:i+2,j-1:j+2],kernel_180).sum())
        if(temp_2>255):
           Convolution[i,j]=255
        elif(temp_2<0):
           Convolution[i,j]=0
        else:
           Convolution[i,j]=temp_2

#Disply Correlation output
cv2.namedWindow('Correlation', cv2.WINDOW_NORMAL)
cv2.imshow('Correlation',Correlation)

#Display Convolution output
cv2.namedWindow('Convolution', cv2.WINDOW_NORMAL)
cv2.imshow('Convolution',Convolution)

cv2.waitKey(0)
cv2.destroyAllWindows()
