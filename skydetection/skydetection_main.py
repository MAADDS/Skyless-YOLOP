from sky_detector import detector
import cv2
from matplotlib import pyplot as plt 
import os


img = cv2.imread("sample/cabc9045-5a50690f.jpg")[:,:,::-1]
h,w,_ = img.shape
img = img[:int(h/2),:]#int(w/3):int(2*w/3)]
plt.figure(2)
plt.subplot(2,1,1)
plt.imshow(img)

img_sky = detector.get_sky_region_gradient(img)
plt.figure(2)
plt.subplot(2,1,2)
plt.imshow(img_sky)
plt.show()