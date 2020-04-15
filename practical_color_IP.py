# Practical Color Image Processing

import numpy as np
import cv2

#Read color image
img = cv2.imread('images/lena.bmp')

# Get the image's height, width and channels
height, width, channels = img.shape

# Create blank Binary Image
img_binary = np.zeros((height,width,1))

# Create grayscale image
img_grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Print image grayscale

# Implementation without OpenCV Library
# ======================
# Set Threshold
thresh = 150
#Calculate
for i in np.arange(height):
    for j in np.arange(width):
        x = img_grayscale.item(i,j)
        if x >= thresh:
            y = 1
        else :
            y = 0

        img_binary.itemset((i,j,0),int(y))

# Write image
#cv2.imwrite('image_binary.jpg',img_binary)
# Show image
cv2.imshow('image',img_binary)
cv2.waitKey(0)
cv2.destroyAllWindows()
                
    
