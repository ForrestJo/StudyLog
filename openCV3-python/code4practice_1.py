
#pip install opencv-python


#Exercise 1
#图像属性：shape, size, dtype
img = imread('1.jpg')
img.size
img.dtype
img.shape



#Exercise 2
videoCapture = cv2.VideoCapture('g:/py/tree.avi')
fps = videoCapture.get(cv2.CAP_PROP_FPS)
size = (
        int(videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH)), 
        int(videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        )
success, frame = videoCapture.read()
while success and cv2.waitKey(1) == -1:
    cv2.imshow('win', frame)
    success, frame = videoCapture.read()

cv2.destroyAllWindows()



#Exercise 3: 
import cv2
import numpy as np
from scipy import ndimage

img = cv2.imread('g:/py/1.jpg', 0)
kernel_3x3 = np.array([[-1, -1, -1],
                        [-1, 8, -1],
                        [-1, -1, -1]
                        ])

kernel_5x5 = np.array([[-1, -1, -1, -1, -1],
                        [-1, 1, 2, 1, -1],
                        [-1, 2, 4, 2, -1],
                        [-1, 1, 2, 1, -1],
                        [-1, -1, -1, -1, -1]
                        ])

k3 = ndimage.convolve(img, kernel_3x3)
k5 = ndimage.convolve(img, kernel_5x5)
blurred = cv2.GaussianBlur(img, (11, 11), 0)
g_hpf = img - blurred
cv2.imshow('3x3',k3)
cv2.imshow('5x5', k5)
cv2.imshow('g_hpf', g_hpf)
if cv2.waitKey() != 0:
    cv2.destroyAllWindows()
