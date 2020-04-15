#Question - 1
#Mean filter
import numpy as np
import cv2

img = cv2.imread('images/lena.bmp', cv2.IMREAD_GRAYSCALE)

blur = cv2.blur(img, (3,3))
cv2.imwrite('images/filter_mean3x3.jpg', blur)

cv2.imshow('image',blur)
cv2.waitKey(0)
cv2.destroyAllWindows()


