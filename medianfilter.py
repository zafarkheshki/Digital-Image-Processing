import numpy as np
import cv2

img = cv2.imread('images/lena.bmp', cv2.IMREAD_GRAYSCALE)
img_out = img.copy()

height = img.shape[0]
width = img.shape[1]

for i in np.arange(1, height-1):
    for j in np.arange(1, width-1):
        neighbors = []
        for k in np.arange(-1,1):
            for l in np.arange(-1,1):
                a = img.item(i+k, j+l)
                neighbors.append(a)
        neighbors.sort()
        median = neighbors[2]
        b = median
        img_out.itemset((i,j), b)

cv2.imwrite('images/filter_median3x3.jpg', img_out)

cv2.imshow('image',img_out)
cv2.waitKey(0)
cv2.destroyAllWindows()
