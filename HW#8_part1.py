import cv2
import numpy as np

#Read image from file
img = cv2.imread('wrench.png')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
median = cv2.medianBlur(gray,5)
edges = cv2. Canny(median,200,300)  # minVal and max value
cv2.imshow('edges',edges)
kernel = np.ones ((5,5) ,np.uint8)
bw  = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel) # Dilation followed by Erosion
cv2.imshow('bw',bw)
# image fill
im_floodfill  = bw.copy()
# Notice the size needs to be 2 pixels than the image.
h, w = bw.shape[:2]
mask = np.zeros((h+2, w+2), np.uint8)

#Floodfill from point (0,0)
cv2.floodFill(im_floodfill, mask, (0,0), 255);
cv2.imshow('im_floodfill',im_floodfill)
#invert floodfilled image
im_floodfill_inv = cv2.bitwise_not(im_floodfill)
cv2.imshow('im_floodfill_inv',im_floodfill_inv)

# Combine the two images to get the foreground
im_out = bw | im_floodfill_inv
cv2.imshow('im_out',im_out)
# remove small things
bw_mask = cv2.morphologyEx(im_out, cv2.MORPH_OPEN, kernel)
kernel = np.ones ((5,5), np.uint8)
e = cv2.erode(bw_mask,kernel,iterations = 1)
cv2.bitwise_xor(e,bw_mask,bw_mask)
#add the red line
bw = edges.copy()
b,g,r  = cv2.split(img)
tmp = np.invert(bw_mask)

ret,thresh1 = cv2.threshold(tmp,100,1,cv2.THRESH_BINARY)
ret,thresh2 = cv2.threshold(bw_mask,100,255,cv2.THRESH_BINARY)
e = cv2.erode(bw_mask,kernel,iterations = 1)
cv2.bitwise_xor(e,thresh2,thresh2)

r = np.multiply(r.copy(),thresh1)+thresh2
img_out = cv2.merge((b,g,r))



#Display image
cv2.imshow('Result',img_out)
cv2.waitKey(0)              #waits for any keyboard event
