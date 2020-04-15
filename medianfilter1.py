#Question - 1
#Mean filter
import numpy as np
import cv2

img = cv2.imread('images/lena.bmp', cv2.IMREAD_GRAYSCALE)


out_mean = cv2.medianBlur(img, 3)


cv2.imwrite('images/filter_median3x3.jpg', out_mean)

cv2.imshow('image',out_mean)
cv2.waitKey(0)
cv2.destroyAllWindows()


