import cv2
import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
# import pytesseract as pt
import json
my_path=os.getcwd()
img=cv2.imread("D://code//Project//OpenCV//image//540.jpg")
img_shape = img.shape

alpha =1.7
beta=0
m1 = cv2.convertScaleAbs(img,alpha=alpha,beta=beta)
plt.imshow(m1,"gray")
plt.show()
#灰階
m2=cv2.cvtColor(m1,cv2.COLOR_BGR2GRAY)
plt.imshow(m2,"gray")
plt.show()
#侵蝕
m3=cv2.erode(m2, np.ones((10,10)))
#膨脹
# m3=cv2.dilate(m3, np.ones((30,30)))
#二值化
ret ,dst= cv2.threshold(m3,125,255,cv2.THRESH_BINARY)
# cv2.imwrite("roi.bmp",dst)
plt.imshow(dst,"gray")
plt.show()
contours, hierarchy = cv2.findContours(dst,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

cv2.drawContours(img,contours,-1,(255,0,0),10)
# plt.imshow(img,"gray")
# plt.show()
#黑白反轉
# dst = 255 - dst
#模糊化去噪
m4=cv2.medianBlur(dst,5)
#膨脹
# dst=cv2.erode(m4, np.ones((70,70)))
#模糊
# dst=cv2.blur(dst, (10, 10))
#黑白反轉
dst = 255 - m4

plt.imshow(dst,"gray")
plt.show()
