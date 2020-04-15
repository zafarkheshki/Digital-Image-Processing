import numpy as np
import cv2

img = cv2.imread('images/lena.bmp', cv2.IMREAD_GRAYSCALE)

k = np.array(([1/4, -1/4, -1/4], [1/4, 2/4, -1/4], [1/4, 1/4, 1/4]), np.float32)

print(k)
print(type(k))

output = numpy.correlate(img, k)

cv2.imwrite('images/correlated_lena.jpg', output)
cv2.imshow('image',output)

cv2.waitKey(0)
cv2.destroyAllWindows()
