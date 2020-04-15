import numpy as np
import cv2



#reading Image from source
img = cv2.imread('golf.jpg')

#Splitting image to RGB Channels
b,g,r = cv2.split(img)

cv2.imshow('blue range',b)
#cv2.imshow('green range',g)
#cv2.imshow('red range',r)

#Applying Median Blur to Image
median=cv2.medianBlur(b,3)
cv2.imshow('median',median)

#Thresholding image
ret,thresh2 = cv2.threshold(median, 140, 255, cv2.THRESH_BINARY)

cv2.imshow('thresh2', thresh2)

#Implementing Kernel
kernel = np.ones((3,3),np.uint8)
bw = cv2.morphologyEx(thresh2,cv2.MORPH_CLOSE, kernel)
im_floodfill = bw.copy()
h,w = bw.shape[:2]
mask = np.zeros((h+2,w+2), np.uint8)

#Floodfill from point 0,0
cv2.floodFill(im_floodfill,mask,(0,0),255)

#Invert Flodfilled image
im_floodfill_inv = cv2.bitwise_not(im_floodfill)

#Combine 2 images to get foreground
im_out = bw|im_floodfill_inv

bw_mask = im_out.copy()

#remove small things
tmp = np.invert(bw_mask)

#add red line
b,g,r = cv2.split(img)
ret,thresh1=cv2.threshold(tmp,100,1,cv2.THRESH_BINARY)
ret,thresh2=cv2.threshold(bw_mask,100,255,cv2.THRESH_BINARY)
r=np.multiply(r.copy(),thresh1)+thresh2
img_out = cv2.merge((b,g,r))

#Displaying Masked Image
cv2.imshow('RESULTS', img_out)

#Applying Blob Detection

params = cv2.SimpleBlobDetector_Params()

params.minThreshold = 0
params.maxThreshold = 255;

#Filter by Area

params.filterByArea = True
params.minArea = 0.1
#Filer by Circularity
params.filterByCircularity = True
params.minCircularity = 0.001
#Filter by Convexity
params.filterByConvexity = True
params.minConvexity = 0.01
#Filter by Inertia
params.filterByInertia = True
params.minInertiaRatio = 0.1
detector = cv2.SimpleBlobDetector_create(params)

#detect blobs
reversemask = 255-bw_mask
keypoints = detector.detect(reversemask)
im_with_keypoints = cv2.drawKeypoints(img, keypoints, np.array([]), (0,255,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

#Show Blobs
cv2.imshow('Keypoints', im_with_keypoints)

nblobs = len(keypoints)

print ('Number of Balls Detected : ', nblobs)

for x in range(0,len(keypoints)):
  imgdraw=cv2.circle(im_with_keypoints, (np.int(keypoints[x].pt[0]),np.int(keypoints[x].pt[1])), radius=np.int(keypoints[x].size), color=(np.random.randint(0,255),np.random.randint(0,255),np.random.randint(0,255)), thickness=-1)

cv2.imshow('filled',imgdraw)
cv2.waitKey(0)







