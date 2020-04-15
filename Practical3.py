# In this code, we are picking an image, adding gaussian noise to it with mean 0 and different standard deviation values of 15, 30 and 45.
# Then applying Low pass and High pass filters on noisy image and displaying the results and their magnitude spectrums. 

import numpy as np
import cv2

# Importing pyplot as plt from Matplotlib
import matplotlib.pyplot as plt

# Reading image from file
img = cv2.imread('images/lena.bmp',0)

row,col = img.shape
mean = 0
sigma15 = 15
gaussian_noise_sigma15 = img+np.uint8(np.random.normal(mean,sigma15,(row,col)))

sigma30 = 30
gaussian_noise_sigma30 = img+np.uint8(np.random.normal(mean,sigma30,(row,col)))

sigma45 = 45
gaussian_noise_sigma45 = img+np.uint8(np.random.normal(mean,sigma45,(row,col)))

img_float32_sigma15 = np.float32(gaussian_noise_sigma15)
img_float32_sigma30 = np.float32(gaussian_noise_sigma30)
img_float32_sigma45 = np.float32(gaussian_noise_sigma45)

# DFT, shift and magnitude spectrum at sigma equals 15
dft_sigma15 = cv2.dft(img_float32_sigma15, flags = cv2.DFT_COMPLEX_OUTPUT)
dft_shift_sigma15 = np.fft.fftshift(dft_sigma15)
magnitude_spectrum_sigma15 = 20*np.log(cv2.magnitude(dft_shift_sigma15[:,:,0],dft_shift_sigma15[:,:,1]))

# DFT, shift and magnitude spectrum at sigma equals 30
dft_sigma30 = cv2.dft(img_float32_sigma30, flags = cv2.DFT_COMPLEX_OUTPUT)
dft_shift_sigma30 = np.fft.fftshift(dft_sigma30)
magnitude_spectrum_sigma30 = 20*np.log(cv2.magnitude(dft_shift_sigma30[:,:,0],dft_shift_sigma30[:,:,1]))

# DFT, shift and magnitude spectrum at sigma equals 45
dft_sigma45 = cv2.dft(img_float32_sigma45, flags = cv2.DFT_COMPLEX_OUTPUT)
dft_shift_sigma45 = np.fft.fftshift(dft_sigma45)
magnitude_spectrum_sigma45 = 20*np.log(cv2.magnitude(dft_shift_sigma45[:,:,0],dft_shift_sigma45[:,:,1]))

###########################################

rows, cols =img.shape
crow,ccol = int(rows/2), int(cols/2)   # Center
# Low Pass Filter Mask
# Creates a mask, Center is 1, remaining all are 0
mask_L = np.zeros((rows,cols, 2), np.uint8)
mask_L[(crow)-60:(crow)+60, (ccol)-60:(ccol)+60] = 1

# Creates a mask, Center is 1, remaining all are 0
mask1_L = np.zeros((rows,cols, 2), np.uint8)
mask1_L[(crow)-60:(crow)+60, (ccol)-60:(ccol)+60] = 1

# Creates a mask, Center is 1, remaining all are 0
mask2_L = np.zeros((rows,cols, 2), np.uint8)
mask2_L[(crow)-60:(crow)+60, (ccol)-60:(ccol)+60] = 1

########################################
# High Pass Filter Mask
# Creates a mask, Center is 0, remaining all are 1s
mask_H = np.ones((rows,cols, 2), np.uint8)
mask_H[(crow)-30:(crow)+30, (ccol)-30:(ccol)+30] = 0

# Creates a mask, Center is 0, remaining all are 1s
mask1_H = np.ones((rows,cols, 2), np.uint8)
mask1_H[(crow)-30:(crow)+30, (ccol)-30:(ccol)+30] = 0

# Create a mask, Center is 0, remaining all are 1s
mask2_H = np.ones((rows,cols, 2), np.uint8)
mask2_H[(crow)-30:(crow)+30, (ccol)-30:(ccol)+30] = 0


# For Sigma = 15
# Applying Low Pass Filter
fshift_sigma15_L= dft_shift_sigma15*mask_L
magnitude_spectrum_sigma15_LPF = 20*np.log(cv2.magnitude(fshift_sigma15_L[:,:,0],fshift_sigma15_L[:,:,1]))

#Inverse DFT(converting to spatial domain)
f_ishift_sigma15_L = np.fft.ifftshift(fshift_sigma15_L)
img_back1_L = cv2.idft(f_ishift_sigma15_L)
img_back1_L = cv2.magnitude(img_back1_L[:,:,0],img_back1_L[:,:,1])

#Applying High Pass Filter
fshift_sigma15_H=fshift_sigma15_L*mask_H
magnitude_spectrum_sigma15_HPF = 20*np.log(cv2.magnitude(fshift_sigma15_H[:,:,0],fshift_sigma15_H[:,:,1]))

#Inverse DFT(converting to spatial domain)
f_ishift_sigma15_H = np.fft.ifftshift(fshift_sigma15_H)
img_back1_H = cv2.idft(f_ishift_sigma15_H)
img_back1_H = cv2.magnitude(img_back1_H[:,:,0],img_back1_H[:,:,1])


# For Sigma = 30
# Applying Low Pass Filter
fshift_sigma30_L= dft_shift_sigma30*mask1_L
magnitude_spectrum_sigma30_LPF = 20*np.log(cv2.magnitude(fshift_sigma30_L[:,:,0],fshift_sigma30_L[:,:,1]))

# Inverse DFT(converting to spatial domain)
f_ishift_sigma30_L = np.fft.ifftshift(fshift_sigma30_L)
img_back2_L = cv2.idft(f_ishift_sigma30_L)
img_back2_L = cv2.magnitude(img_back2_L[:,:,0],img_back2_L[:,:,1])

#Applying High Pass Filter
fshift_sigma30_H=fshift_sigma30_L*mask1_H
magnitude_spectrum_sigma30_HPF = 20*np.log(cv2.magnitude(fshift_sigma30_H[:,:,0],fshift_sigma30_H[:,:,1]))

#Inverse DFT(converting to spatial domain)
f_ishift_sigma30_H = np.fft.ifftshift(fshift_sigma30_H)
img_back2_H = cv2.idft(f_ishift_sigma30_H)
img_back2_H = cv2.magnitude(img_back2_H[:,:,0],img_back2_H[:,:,1])


# For Sigma = 45
# Applying Low Pass Filter
fshift_sigma45_L= dft_shift_sigma45*mask2_L
magnitude_spectrum_sigma45_LPF = 20*np.log(cv2.magnitude(fshift_sigma45_L[:,:,0],fshift_sigma45_L[:,:,1]))

# Inverse DFT(converting to spatial domain)
f_ishift_sigma45_L = np.fft.ifftshift(fshift_sigma45_L)
img_back3_L = cv2.idft(f_ishift_sigma45_L)
img_back3_L = cv2.magnitude(img_back3_L[:,:,0],img_back3_L[:,:,1])

# Applying High Pass Filter
fshift_sigma45_H=fshift_sigma45_L*mask2_H
magnitude_spectrum_sigma45_HPF = 20*np.log(cv2.magnitude(fshift_sigma45_H[:,:,0],fshift_sigma45_H[:,:,1]))

#Inverse DFT(converting to spatial domain)
f_ishift_sigma45_H = np.fft.ifftshift(fshift_sigma45_H)
img_back3_H = cv2.idft(f_ishift_sigma45_H)
img_back3_H = cv2.magnitude(img_back3_H[:,:,0],img_back3_H[:,:,1])

# Output Images for Sigma = 15
plt.figure(1)
plt.subplot(321),plt.imshow(gaussian_noise_sigma15, cmap = 'gray')
plt.title('NoisyImage_Sigma15'), plt.xticks([]),plt.yticks([])
plt.subplot(322),plt.imshow(magnitude_spectrum_sigma15, cmap = 'gray')
plt.title('Magnitude_Spectrum_NoisyImg_Sigma15'), plt.xticks([]),plt.yticks([])
plt.subplot(323),plt.imshow(img_back1_L, cmap = 'gray')
plt.title('LPF_Sigma15 '), plt.xticks([]),plt.yticks([])
plt.subplot(324),plt.imshow(magnitude_spectrum_sigma15_LPF, cmap = 'gray')
plt.title('Magnitude_Spectrum_LPF_Sigma15 '), plt.xticks([]),plt.yticks([])
plt.subplot(325),plt.imshow(img_back1_H, cmap = 'gray')
plt.title('HPF_Sigma15 '), plt.xticks([]),plt.yticks([])
plt.subplot(326),plt.imshow(magnitude_spectrum_sigma15_HPF, cmap = 'gray')
plt.title('Magnitude_Spectrum_HPF_Sigma15 '), plt.xticks([]),plt.yticks([])



# Output Images for Sigma = 30
plt.figure(2)
plt.subplot(321),plt.imshow(gaussian_noise_sigma30, cmap = 'gray')
plt.title('NoisyImage_Sigma30'), plt.xticks([]),plt.yticks([])
plt.subplot(322),plt.imshow(magnitude_spectrum_sigma30, cmap = 'gray')
plt.title('Magnitude_Spectrum_NoisyImg_Sigma30'), plt.xticks([]),plt.yticks([])
plt.subplot(323),plt.imshow(img_back2_L, cmap = 'gray')
plt.title('LPF_Sigma30 '), plt.xticks([]),plt.yticks([])
plt.subplot(324),plt.imshow(magnitude_spectrum_sigma30_LPF, cmap = 'gray')
plt.title('Magnitude_Spectrum_LPF_Sigma30 '), plt.xticks([]),plt.yticks([])
plt.subplot(325),plt.imshow(img_back2_H, cmap = 'gray')
plt.title('HPF_Sigma30 '), plt.xticks([]),plt.yticks([])
plt.subplot(326),plt.imshow(magnitude_spectrum_sigma30_HPF, cmap = 'gray')
plt.title('Magnitude_Spectrum_HPF_Sigma30 '), plt.xticks([]),plt.yticks([])

# Output Images for Sigma = 45
plt.figure(3)
plt.subplot(321),plt.imshow(gaussian_noise_sigma45, cmap = 'gray')
plt.title('NoisyImage_Sigma45'), plt.xticks([]),plt.yticks([])
plt.subplot(322),plt.imshow(magnitude_spectrum_sigma45, cmap = 'gray')
plt.title('Magnitude_Spectrum_NoisyImg_Sigma45'), plt.xticks([]),plt.yticks([])
plt.subplot(323),plt.imshow(img_back3_L, cmap = 'gray')
plt.title('LPF_Sigma45 '), plt.xticks([]),plt.yticks([])
plt.subplot(324),plt.imshow(magnitude_spectrum_sigma45_LPF, cmap = 'gray')
plt.title('Magnitude_Spectrum_LPF_Sigma45 '), plt.xticks([]),plt.yticks([])
plt.subplot(325),plt.imshow(img_back3_H, cmap = 'gray')
plt.title('HPF_Sigma45 '), plt.xticks([]),plt.yticks([])
plt.subplot(326),plt.imshow(magnitude_spectrum_sigma45_HPF, cmap = 'gray')
plt.title('Magnitude_Spectrum_HPF_Sigma45 '), plt.xticks([]),plt.yticks([])


plt.show()

