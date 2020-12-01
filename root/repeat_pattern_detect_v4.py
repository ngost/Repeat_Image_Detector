from matplotlib import pyplot as plt
import numpy as np
import cv2
import math
from scipy.stats import norm
from vector_similarity import euclidean,different
from proto_repeat_detector import norm_list,avg_list
import scipy.stats as stats
import numpy as np
from skimage import io
from matplotlib import pyplot as plt



img_name = 'img/r19.jpg'
img = cv2.imread(img_name)
w, h,c = img.shape
print(w)
if w > 1000 :
    print(w)
    img = cv2.resize(img, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
try:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
except ValueError as e:
    gray = img
gray = cv2.blur(gray, (3, 3))
sift = cv2.xfeatures2d.SIFT_create()
kp, des = sift.detectAndCompute(gray, None)

#img = cv2.drawKeypoints(gray, kp, img,)



positionX = []
positionY = []
i = 0
small_feature = []

avg_size = 0.0
for point in kp:
    temp = point.size
    avg_size += temp

avg_size = avg_size/len(kp)
scale = []
print("size_avg : "+str(avg_size))
# for point in kp:
#     temp = point.size
#     scale.append(point.size)
#     #temp = point.pt[0]
#     if temp<avg_size:
# #        print(temp)
# #        print(temp)
#         small_feature.append(i)
#     i += 1
# #    positionX.append(temp)
#   #  temp = point.angle
#     #temp = point.pt[1]
#  #   print(temp)
#  #   positionY.append(temp)
# plt.hist(scale,bins=len(scale))
# plt.show()
# print(small_feature)
height, width, channels = img.shape

i = 0
sum_data = 0
kp_num = len(kp)
des_sum = []

# for del_e in reversed(small_feature):
#     des = np.delete(des,del_e,0)


while i < 128:
    for descriptor in des:
        sum_data = sum_data + descriptor[i]

    sum_data = sum_data/kp_num
    des_sum.append(sum_data)
    #print(sum_data)
    sum_data = 0
    i += 1

#print(des_sum) #des_sum = descriptor들의 평균
#print(des[0])

euclidean_result = []
for descriptor in des:
    euclidean_result.append(euclidean(des_sum,descriptor))


print(euclidean_result)
bins_num = len(set(euclidean_result))
print(bins_num)
#bins_num = 30
#bins_num = max(euclidean_result)-min(euclidean_result)

euclidean_result = norm_list(euclidean_result, max(euclidean_result),min(euclidean_result))
#norm_euc = euclidean_result
#euclidean_result.sort()
print(euclidean_result)

count, bins, ignored = plt.hist(euclidean_result,bins_num,color='red',density=True)
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
print(bins)
#bins를 조작해서,
bins2 = []
for bins_e in bins:
    bins2.append(-bins_e)
bins2.sort(reverse=False)

bins2.extend(bins)
print("bins3")
print(bins2)

plt.plot(bins2, 1/(sigma * np.sqrt(2 * np.pi)) *	np.exp( - (bins2 - mu)**2 / (2 * sigma**2)), linewidth=1, color='b')


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

#뜬금 생각

test = 1/(sigma * np.sqrt(2 * np.pi)) * np.exp(-(bins2- mu)**2 / (2 * sigma**2))
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

#print(count[2])
count = count.tolist()
count2 = []
for count2_e in range(len(count)+2):
    count2.append(0)

count2.extend(count)
test = test.tolist()
print(count2)
print(len(count2))
print(test)
print(len(test))
real_result = euclidean(count2,test)
real_result2 = different(count2,test)

print(1/(1+real_result))
print(real_result2/len(test))