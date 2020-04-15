#Question - 1
#Mean filter
import numpy as np
import cv2

img = cv2.imread('images/lena.bmp', cv2.IMREAD_GRAYSCALE)


kernel = np.matrix('1 1 1; 1 1 1; 1 1 1')/9
out_mean = cv2.filter2D(img, -1, kernel)


cv2.imwrite('images/filter_mean3x3.jpg', out_mean)

cv2.imshow('image',out_mean)
cv2.waitKey(0)
cv2.destroyAllWindows()


