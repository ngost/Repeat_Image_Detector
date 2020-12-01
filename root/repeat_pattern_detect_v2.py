from matplotlib import pyplot as plt
import numpy as np
import cv2
import math
from scipy.stats import norm
from vector_similarity import euclidean
from proto_repeat_detector import norm_list,avg_list
import scipy.stats as stats
import numpy as np
from skimage import io
from matplotlib import pyplot as plt


bin_num = 50
img_name = 'img/g1.jpeg'
img = io.imread(img_name)
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
gray = cv2.blur(gray, (3, 3))
sift = cv2.xfeatures2d.SIFT_create()
kp, des = sift.detectAndCompute(gray, None)
#img = cv2.drawKeypoints(gray, kp, img,)



positionX = []
positionY = []
# for point in kp:
#     temp = point.size
#     #temp = point.pt[0]
#     print(temp)
#     positionX.append(temp)
#     temp = point.angle
#     #temp = point.pt[1]
#     print(temp)
#     positionY.append(temp)
height, width, channels = img.shape

i = 0
sum_data = 0
kp_num = len(kp)
des_sum = []
while i < 128:
    for descriptor in des:
        sum_data = sum_data + descriptor[i]

    sum_data = sum_data/kp_num
    des_sum.append(sum_data)
    print(sum_data)
    sum_data = 0
    i += 1

print(des_sum) #des_sum = descriptor들의 평균
print(des[0])

euclidean_result = []
for descriptor in des:
    euclidean_result.append(euclidean(des_sum,descriptor))

print(euclidean_result)
#norm_euc = norm_list(euclidean_result, max(euclidean_result),0)
norm_euc = euclidean_result

count, bins, ignored = plt.hist(norm_euc, bin_num,normed=True,color='red')
# sample들을 이용해서 Gaussian Distribution의 shape을 재구축해서 line으로 그린다.

sigma = np.std(norm_euc)
mu = np.mean(norm_euc)
print(sigma)
print(mu)
args = plt.plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) *	np.exp( - (bins - mu)**2 / (2 * sigma**2) ), linewidth=2, color='b')

#plt.hist(euclidean_result, facecolor='red',bins=50)  # arguments are passed to np.histogram
plt.title(img_name)
plt.show()
#avg_list(norm_e)
#des1 = des[]
#des2 = des[2]
#print(des1)
#print(des2)

result = 1/(sigma * np.sqrt(2 * np.pi)) * np.exp(-(bins - mu)**2 / (2 * sigma**2))

print(result)
print(count)

i = 0
sum_error = 0

while i<bin_num :
    sum_error += (( (result[i]+result[i+1]) / 2) - count[i]) ** 2
    i += 1


print("에러율 : "+str((sum_error/bin_num)))
# print(bins)
#
# print(max(result))
#
# result_max = max(result)
# bin_new = []
# for bin_ele in bins:
#     bin_new.append(result_max/bin_ele)
#
# print(bin_new)
# print(bin_new[0])