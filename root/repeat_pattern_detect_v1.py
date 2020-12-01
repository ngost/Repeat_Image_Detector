from matplotlib import pyplot as plt
import numpy as np
import cv2
import math
from scipy.stats import norm
from vector_similarity import euclidean
from proto_repeat_detector import norm_list

def avg_list(lst):
    lst_sum = sum(lst)
    return lst_sum/len(lst)


img_name = 'img/g1.jpeg'
img = cv2.imread(img_name)
gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
gray = cv2.blur(gray, (3, 3))
sift = cv2.xfeatures2d.SIFT_create()
kp, des = sift.detectAndCompute(gray, None)
height, width, channels = img.shape
i = 0
sum_data = 0
kp_num = len(kp)
des_avg = []
while i < 128:
    for descriptor in des:
        sum_data = sum_data + descriptor[i]

    sum_data = sum_data/kp_num
    des_avg.append(round(sum_data))
    sum_data = 0
    i += 1

euclidean_result = []
#print(des_avg)

for descriptor in des:
    euclidean_result.append(euclidean(des_avg, descriptor))

#정규화
bins_num = max(euclidean_result)-min(euclidean_result)
bins_num = len(set(euclidean_result))

norm_data = norm_list(euclidean_result,max(euclidean_result),min(euclidean_result))

plt.hist(norm_data, facecolor='red',bins=bins_num,normed=False)  # arguments are passed to np.histogram




#정규분포 그리기

mean = avg_list(norm_data)# 평균
std = np.std(norm_data, ddof=1) # 표준편차
data = np.random.normal(mean, std, len(norm_data))
#plt.hist(data, bins=bins_num,facecolor='blue',normed=False) # 나누는 구간 개수 (100개 정도로 더 잘게 나누어 보라는 의미)
plt.show()
#print(euclidean_result)
print(norm_data)
print(data)

norm_data.sort()
data.sort()
result_final = euclidean(norm_data,data)

print(1/(result_final+1))

