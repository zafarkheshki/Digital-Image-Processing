# Question - 3
# Laplacian

import numpy as np
import cv2

#Read image from file
img = cv2.imread('images/lena.bmp', cv2.IMREAD_GRAYSCALE)
laplacian = img.copy()

# getting height and width of the image
width,height = img.shape[:2]

kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])

print(img[0:3,0:3])
for i in range(1,width-2):
    for j in range (1,height-2):
       temp = int(np.multiply(img[i-1:i+2,j-1:j+2],kernel).sum()/4);
       if(temp>255):
           laplacian[i,j]=255
       elif(temp<0):
           laplacian[i,j]=0
       else:
           laplacian[i,j]=temp;
           
#Disply Laplacian output
cv2.namedWindow('Laplacian', cv2.WINDOW_NORMAL)
cv2.imshow('Laplacian',laplacian)

cv2.waitKey(0)
cv2.destroyAllWindows()
