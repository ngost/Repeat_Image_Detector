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



img_name = 'img/noise_img.jpg'
img = io.imread(img_name)
gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
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
    des_sum.append(round(sum_data))
    #print(sum_data)
    sum_data = 0
    i += 1

#print(des_sum) #des_sum = descriptor들의 평균
#print(des[0])

euclidean_result = []
for descriptor in des:
    euclidean_result.append(round(euclidean(des_sum,descriptor)))


print(euclidean_result)
bins_num = len(set(euclidean_result))
print(bins_num)
#bins_num = 30
#bins_num = max(euclidean_result)-min(euclidean_result)

euclidean_result = norm_list(euclidean_result, max(euclidean_result),min(euclidean_result))
#norm_euc = euclidean_result
#euclidean_result.sort()
print(euclidean_result)

count, bins, ignored = plt.hist(euclidean_result,bins_num,color='red',normed=True)
#print("bin")
#print(count)
#print(len(count))
# sample들을 이용해서 Gaussian Distribution의 shape을 재구축해서 line으로 그린다.
#plt.show()
sigma = np.std(euclidean_result)
mu = np.mean(euclidean_result)

# sigma = np.std(list(set(count)))
# mu = np.mean(list(set(count)))


print(sigma)
print(mu)

plt.plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) *	np.exp( - (bins - mu)**2 / (2 * sigma**2)), linewidth=1, color='b')


#plt.hist(euclidean_result, facecolor='red',bins=50)  # arguments are passed to np.histogram
plt.title(img_name)
plt.show()
#avg_list(norm_e)
#des1 = des[]
#des2 = des[2]
#print(des1)
#print(des2)

#print(len(set(euclidean_result)))
#result = 1/(sigma * np.sqrt(2 * np.pi)) * np.exp(-(bins - mu)**2 / (2 * sigma**2))
test_index = []

for i in range(bins_num):
    test_index.append(i/bins_num)
#print(test_index)
test = 1/(sigma * np.sqrt(2 * np.pi)) * np.exp(-(bins- mu)**2 / (2 * sigma**2))
#test
#print("test")

#test : 실제 euclidean_data에서 값이 존재하는 index들에 대해, 연속정규분포 함수에 식을 대입하여 얻은 정규분포 데이터 (정규분포에서 나온 비교대상)
#hist_pdf : index 값에 따른 histgram의 pdf 데이터들
# print(test)
# print(list(euclidean_index))
# print(euclidean_result)
#
# print(result)
# print(count)
#
# print(len(test))
#
# i = 0
# sum_error = 0
# while i<bin_num :
#     sum_error += (( (result[i]+result[i+1]) / 2) - count[i]) ** 2
#     i += 1


#print("에러율 : "+str((sum_error/bin_num)))
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

#print("범위 :")
#print(len(count))
#print(len(test))

#print("에러율2:")
#count.sort()
#test.sort()
#count는
print(count)
i = 0
del_list = []
for count_e in count:
    if count_e == 0.0:
        del_list.append(i)
    i += 1

print(del_list)
#print(count[2])
count = count.tolist()
print(count)
test = test.tolist()


for del_e in reversed(del_list):
    count.pop(del_e)
    test.pop(del_e)

count.sort()
test.sort()
real_result = euclidean(count,test)
# for count_e,test_e in zip(count,test):
#     val = (count_e - test_e) ** 2
#     if val>1:
#         print(val)
#         print(count_e)
#         print(test_e)
#         print("")
# print(1/(1+real_result))
i=0

sum_error = 0

print(1/(1+real_result))