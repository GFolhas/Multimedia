import matplotlib.pyplot as plt
import matplotlib.colors as clr
import numpy as np
import cv2  #pip3 install opencv-python
import scipy.fftpack as fft
import math as m    

nl = 16 #deve ser multiplo de 16!
nc = 16 #deve ser multiplo de 16!
colors = 3

images = "imagens/barn_mountains.bmp"
img = plt.imread(images)


print("Downsampling 4:2:0 using no interpolation filter")
print()
scaleX = 0.5
scaleY = 0.5

stepX = int(1//scaleX)
stepY = int(1//scaleY)

dsImg = img[::stepY, ::stepX, :]

fig = plt.figure(figsize=(10, 10))
fig.add_subplot(1, 2, 1)
plt.imshow(img)
plt.title('Original')
plt.axis('image')

fig.add_subplot(1, 2, 2)
plt.imshow(dsImg)
plt.title('downsampled 4:2:0 sx = 0.5, sy = 0.5')
plt.axis('image')
plt.show()

print()
print("Downsampling 4:2:0 using openCv with interpolation filter")
print()

dsImgInterp = cv2.resize(img, None, fx=scaleX, fy=scaleY, interpolation=cv2.INTER_LINEAR)

fig = plt.figure(figsize=(10, 10))
fig.add_subplot(1, 2, 1)
plt.imshow(img)
plt.title('Original')
plt.axis('image')

fig.add_subplot(1, 2, 2)
plt.imshow(dsImgInterp)
plt.title('downsampled 4:2:0 sx = 0.5, sy = 0.5 interpolated')
plt.axis('image')
plt.show()

print()
print("Upsampling with repetitions")
print()

fig = plt.figure(figsize=(20, 20))

usImg = np.repeat(dsImg, stepX, axis=1)
l, c, p = usImg.shape
usImg = np.repeat(usImg, stepY, axis=0)

fig.add_subplot(1, 4, 1)
plt.imshow(img)
plt.title('original')
plt.axis('image')

fig.add_subplot(1, 4, 2)
plt.imshow(dsImg)
plt.title('downsampled 4:2:0 no interp')
plt.axis('image')

fig.add_subplot(1, 4, 3)
plt.imshow(usImg)
plt.title('upsampled with repetitions')
plt.axis('image')
plt.show()

print()
print("dsImg size = ", dsImg.shape)
print("usImg size = ", usImg.shape)

print()
print("Upsampling with interpolation")
print()

fig = plt.figure(figsize=(20, 20))

usImg = cv2.resize(dsImg, None, fx=stepX, fy=stepY, interpolation=cv2.INTER_LINEAR)
fig.add_subplot(1, 4, 1)
plt.imshow(img)
plt.title('original')
plt.axis('image')

fig.add_subplot(1, 4, 2)
plt.imshow(dsImg)
plt.title('downsampled 4:2:0 no interp')
plt.axis('image')

fig.add_subplot(1, 4, 3)
plt.imshow(usImg)
plt.title('upsampled with interpolation')
plt.axis('image')
plt.show()

print()
print("dsImg size = ", dsImg.shape)
print("usImg size = ", usImg.shape)