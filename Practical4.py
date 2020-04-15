import tkinter as tk
from tkinter import *
from PIL import ImageTk, Image
import cv2
from tkinter import filedialog
import numpy as np
import matplotlib.pyplot as plt

import os

os.system('''/usr/bin/osascript -e 'tell app "Finder" to set frontmost of process "Python" to true' ''')


class Assignment(tk.Frame):
   def Apply(self):
       global noise

       self.img_new = Image.eval(self.img, lambda x: x * self.var_contrast.get() + self.var_brightness.get())

       if (self.var_gs.get() == 0 and self.var_bn.get() == 0 and self.var_noise.get() == 0 and self.var_gnoise.get() == 0):
           self.canvas.image = ImageTk.PhotoImage(self.img_new)
       elif self.var_gs.get() == 1:

           self.gray = 0.2989 * np.array(self.img)[:,:,0] + 0.5870 * np.array(self.img)[:,:,1] + 0.1140 * np.array(self.img)[:,:,2]
           self.gray = Image.fromarray(self.gray)
           self.canvas.image = ImageTk.PhotoImage(self.gray)
       elif self.var_bn.get() == 1:

           self.gray = 0.2989 * np.array(self.img)[:,:,0] + 0.5870 * np.array(self.img)[:,:,1] + 0.1140 * np.array(self.img)[:,:,2]
           self.gray[self.gray < 127] = 0 
           self.gray[self.gray >= 127] = 255 
           self.binary = self.gray
           self.binary = Image.fromarray(self.binary)
           self.canvas.image = ImageTk.PhotoImage(self.binary)

       # Adds Salt & Pepper noise
       elif self.var_noise.get() == 1 and self.var_gnoise.get() == 0:
           prob = 0.05
           image = np.array(self.img)  # Implements Salt and Pepper Noise

           output = np.zeros(image.shape,np.uint8)
           thres = 1 - prob 
           for i in range(image.shape[0]):
               for j in range(image.shape[1]):
                   rdn = np.random.random()
                   if rdn < prob:
                       output[i][j] = 0
                   elif rdn > thres:
                       output[i][j] = 255
                   else:
                       output[i][j] = image[i][j]
           # Returns output
           self.noise = Image.fromarray(output)
           self.canvas.image = ImageTk.PhotoImage(self.noise)

       #Adds Gaussian noise
       elif self.var_gnoise.get() == 1:
           #Implement the Gaussian noise
           row,col,ch = np.shape(self.img)
           gauss = np.random.normal(0,20,(row,col,ch))
           gauss = gauss.reshape(row,col,ch)
           noise = self.img + gauss
           noise = Image.fromarray(noise.astype('uint8'))
           self.canvas.image = ImageTk.PhotoImage(noise)

       self.canvas.create_image(0, 0, image=self.canvas.image, anchor='nw')


   def browse(self):
       global noise
       self.filename = filedialog.askopenfilename()
       self.img = cv2.imread(self.filename, 1)
       self.b, self.g, self.r = cv2.split(self.img)
       self.img1 = cv2.merge((self.r, self.g, self.b))
       self.img = Image.fromarray(self.img1)
       self.img2=Image.fromarray(self.img1)
       self.img = self.img2.resize((512,512))
       noise = self.img
       self.canvas.image = ImageTk.PhotoImage(self.img)
       self.canvas.create_image(0, 0, image=self.canvas.image, anchor='nw')

   def lowpass(self):
       global noise
       noise_n = np.array(noise).astype(np.uint8)
       def lowpass_0(img,cp):
           print(img.shape)
           rows, cols = img.shape
           crow, ccol = np.int(rows/2), np.int(cols/2)
           mask = np.zeros((rows, cols, 2), np.uint8)
           mask[(crow)-cp:(crow)+cp,(ccol)-cp:(ccol)+cp] = 1
           return mask

       def lowpass_1(img):
           img_float32 = np.float32(img)

           dft = cv2.dft(img_float32, flags = cv2.DFT_COMPLEX_OUTPUT)
           dft_shift = np.fft.fftshift(dft)
           magnitude_spectrum_img = 20*np.log(cv2.magnitude(dft_shift[:,:,0],dft_shift[:,:,1]))

           mask_lp = lowpass_0(img, 60);

           f_shift_lp = dft_shift * mask_lp; 
           f_ishift_lp = np.fft.ifftshift(f_shift_lp)
           img_back_lp = cv2.idft(f_ishift_lp, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT)
           return img_back_lp

       r, g, b = np.array(noise_n)[:,:,0], np.array(noise_n)[:,:,1], np.array(noise_n)[:,:,2]
       R = lowpass_1(r)
       G = lowpass_1(g)
       B = lowpass_1(b)

       rgb = (R[..., np.newaxis], G[..., np.newaxis], B[..., np.newaxis])
       img = np.concatenate(rgb, axis=-1)
       
       img_back = Image.fromarray(np.array(img).astype(np.uint8))
       self.canvas.image = ImageTk.PhotoImage(img_back)
       self.canvas.create_image(0, 0, image=self.canvas.image, anchor='nw')


   def highpass(self):
        global noise
        noise_n = np.array(noise).astype(np.uint8)
        def highpass_0(img,cp):
            print(img.shape)
            rows, cols = img.shape
            crow, ccol = np.int(rows/2), np.int(cols/2)
            mask = np.ones((rows, cols, 2), np.uint8)
            mask[(crow)-cp:(crow)+cp,(ccol)-cp:(ccol)+cp] = 0
            return mask

        def highpass_1(img):
            img_float32 = np.float32(img)

            dft = cv2.dft(img_float32, flags = cv2.DFT_COMPLEX_OUTPUT)
            dft_shift = np.fft.fftshift(dft)
            magnitude_spectrum_img = 20*np.log(cv2.magnitude(dft_shift[:,:,0],dft_shift[:,:,1]))

            mask_hp = highpass_0(img, 5);

            f_shift_hp = dft_shift * mask_hp; 
            f_ishift_hp = np.fft.ifftshift(f_shift_hp)
            img_back_hp = cv2.idft(f_ishift_hp, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT)
            return img_back_hp
        
        r, g, b = np.array(noise_n)[:,:,0], np.array(noise_n)[:,:,1], np.array(noise_n)[:,:,2]
        R = highpass_1(r)
        G = highpass_1(g)
        B = highpass_1(b)
        
        rgb = (R[..., np.newaxis], G[..., np.newaxis], B[..., np.newaxis])
        img = np.concatenate(rgb, axis=-1)
        img_back = Image.fromarray(np.array(img).astype(np.uint8))
        self.canvas.image = ImageTk.PhotoImage(img_back)
        self.canvas.create_image(0, 0, image=self.canvas.image, anchor='nw')
        

   def medfil(self):
       global noise
       noise = cv2.medianBlur(np.array(noise).astype(np.uint8),5)
       noise = Image.fromarray(noise)
       self.canvas.image = ImageTk.PhotoImage(noise)
       self.canvas.create_image(0, 0, image=self.canvas.image, anchor='nw')


   def __init__(self, root):
       tk.Frame.__init__(self, root)

       # Browse
       self.btn_browse = tk.Button(root, text="Browse", command=self.browse)
       self.btn_browse.grid(row=0, column=0)

       # Apply
       self.btn_apply = tk.Button(root, text="Apply", command=self.Apply)
       self.btn_apply.grid(row=0, column=1)

       # Low-pass Filter
       self.btn_lowpass = tk.Button(root, text="Lowpass-Filter", command=self.lowpass)
       self.btn_lowpass.grid(row=3, column=4)

       # High-pass Filter
       self.btn_highpass = tk.Button(root, text="Highpass-Filter", command=self.highpass)
       self.btn_highpass.grid(row=5, column=4)

       # Median-Filter
       self.btn_medfil = tk.Button(root, text="Median-Filter", command=self.medfil)
       self.btn_medfil.grid(row=1, column=4)

       # Canvas
       self.canvas = tk.Canvas(root, width=512, height=512)
       self.canvas.grid(row=7, columnspan=2)

       # Grayscale
       self.var_gs = tk.IntVar()
       self.chk = tk.Checkbutton(root, text='Grayscale', variable=self.var_gs)
       self.chk.grid(row=0, column=2)

       # Binary
       self.var_bn = tk.IntVar()
       self.chk = tk.Checkbutton(root, text='Binary', variable=self.var_bn)
       self.chk.grid(row=1, column=2)

       # Salt & Pepper noise
       self.var_noise = tk.IntVar()
       self.chk = tk.Checkbutton(root, text='Add Salt & Pepper Noise', variable=self.var_noise)
       self.chk.grid(row=2, column=2)

       # Gaussian noise
       self.var_gnoise = tk.IntVar()
       self.chk = tk.Checkbutton(root, text='Add Gaussian Noise', variable=self.var_gnoise)
       self.chk.grid(row=3, column=2)

       # Slider brightness
       self.label_brightness = tk.Label(root, text="Brightness")
       self.label_brightness.grid(row=0, column=3)
       self.var_brightness = tk.Scale(root, from_=0, to=100, orient='horizontal')
       self.var_brightness.grid(row=1, column=3)

       # Slider contrast
       self.label_contrast = tk.Label(root, text="Contrast")
       self.label_contrast.grid(row=2, column=3)
       self.var_contrast = tk.Scale(root, from_=1, to=100, orient='horizontal')
       self.var_contrast.grid(row=3, column=3)



# Initialise
root = tk.Tk()
root.wm_title("Assignment")
# Initialise class
Assignment(root).grid()
# Live
root.mainloop()
