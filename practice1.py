import numpy as np
import cv2

#read image from file
img = cv2.imread(lena.bnp)
gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
out=img.copy()
#get width and height
width, height = gray_img.shape[:2]

x=5
y=5
print(gray_image[x-1:x+2,y-1:y+2])
print(gray_image[x-1:x+2,y-1:y+2].sum())
print(int(gray_image[x-1:x+2,y-1:y+2].sum()/9)

for x in range(1, width-2):
      for y in range(1, height-2):
          out[x,y] = int(gray_image[x-1:x+2,y-1:y+2].sum()/9)
      
for x in range(1, width-2):
      for y in range(1, height-2):
          out2[x,y] = np.median(gray_image[x-1:x+2,y-1:y+2])

#display image
      cv2.namedwindow('Mean', cv2.WINDOW_NORMAL)
      cv2.imshow('Mean',out)
#display image
      cv2.namedwindow('Median', cv2.WINDOW_NORMAL)
      cv2.imshow('Median',out)
      cv2.waitKey(0)    #waits for any keyboard event
      
