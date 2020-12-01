from matplotlib import pyplot as plt
import numpy as np
import cv2
import math
from scipy.stats import norm
from vector_similarity import euclidean
from skimage import io
from proto_repeat_detector import avg_list,norm_list
from img_noise import sp_noise

img = cv2.imread('img/lower_variance.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray2 = gray
#gray2 = sp_noise(gray, 0.005)
sift = cv2.xfeatures2d.SIFT_create()
kp, des = sift.detectAndCompute(gray2, None)

plt.imshow(gray2)
plt.show()
height, width, channels = img.shape

i = 0
sum_data = 0
kp_num = len(kp)
positionX = []
positionY = []
for point in kp:
    positionX.append(round(point.pt[0]))
    positionY.append(round(point.pt[1]))

print(positionX)
print(positionY)

# ex_listX = list(set(positionX))
# ex_listX.sort()
#
# ex_listY = list(set(positionY))
# ex_listY.sort()
# positionX.sort()
# positionY.sort()
ex_listX = positionX
ex_listY = positionY
print("..variance")
print(np.var(positionX))
print(np.var(positionY))
print("..avg")
print(avg_list(positionX))
print(avg_list(positionY))

ex_listX = norm_list(ex_listX,width,0)
ex_listY = norm_list(ex_listY,height,0)

print("..norm variance")
print(np.var(ex_listX))
print(np.var(ex_listY))