from matplotlib import pyplot as plt
import numpy as np
import cv2
import math
from scipy.stats import norm

def get_entropy(pixel_size, nphist):
    hist = nphist[0].tolist()
    pdf_list = []
    w_count = {}

    for lst in hist:
        try:
            w_count[lst] += 1
        except:
            w_count[lst] = 1
    #print(w_count)

    key_list = w_count.keys()

    for key in key_list:
        pdf_list.append(w_count[key]/(pixel_size-1))

    print(pdf_list)

    #real = sum(hist)
#    feature_number = real
    #hist = [1]
    #print(hist)
    entropy = 0
    percent = 0
    zero = 0
    for data in pdf_list:
        entropy = entropy + data*math.log2(data)
        #percent = percent + (data/pixel_size)

#    entropy = entropy + (zero/pixel_size)*math.log2(zero/pixel_size)

    #print(pixel_size)
    #print(len(hist)+1)
    #print(pixel_size/(len(hist)+1))

    #print(sum(hist))

#    percent = percent + zero/pixel_size

    #print(percent)
    return -entropy


def get_vector_length(ls):
    sum_val = 0

    for val in ls:
        sum_val = sum_val + val ** 2

    result_val = math.sqrt(sum_val)
    return result_val

img = cv2.imread('img/r1.jpg')
gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
sift = cv2.xfeatures2d.SIFT_create()
kp,des = sift.detectAndCompute(gray, None)
img = cv2.drawKeypoints(gray, kp, img)
feature_count = 0
angle_max = 360
angle = []
size = []
etc = []
positionX = []
positionY = []
for point in kp:
    positionX.append(point.pt[0])
    positionY.append(point.pt[1])
    feature_count = feature_count + 1
    temp = point.size
    #temp = point.pt[0]
    #size.append(temp)
    size.append(round(temp,1))
    temp = point.angle
    #temp = point.pt[1]
    #angle.append(temp)
    angle.append(round(temp,1))
    #print(temp)

print(feature_count)
height, width, channels = img.shape

##plt.hist(size, facecolor='blue',bins='auto')  # arguments are passed to np.histogram
##plt.title("Entropy size")
##plt.show()

##plt.hist(angle, facecolor='blue',bins=360)  # arguments are passed to np.histogram
##plt.title("Entropy angle")
##plt.show()

#plt.hist(angle, facecolor='red',bins = height,range=[0,height+1])  # arguments are passed to np.histogram
#plt.title("Entropy Y")
#plt.show()

#feature_num = len(set(positionX))
##nphist_angle = np.histogram(angle, bins='auto')
##print('angle_entropy : ' + str(get_entropy(feature_count, nphist_angle)))

##nphist_size = np.histogram(size, bins='auto')
##print('size_entropy : ' + str(get_entropy(feature_count, nphist_size)))

#feature_num = len(set(positionY))
#nphisty = np.histogram(positionY, bins=range(height))
#print(nphisty[0].tolist())
#print('y entropy : ' + str(get_entropy(height, nphisty)))


#des_list = des.tolist()
des_list2 = list(des)
desc_vectors = []
#print(des_list)
for array in des_list2:
    print(list(array))
    desc_vectors.append(array[0])

print(desc_vectors)
print("feature number :"+str(feature_count))
print("vectors number :"+str(len(des)))
#print(desc_vectors)
#print(len(desc_vectors))

binOfdes = len(des)
#plt.hist(position[], facecolor='red')  # arguments are passed to np.histogram
plt.scatter(positionX,positionY,s=1)

plt.title("descriptor_1")
plt.show()

#print(size)

#print(len(angle))
#print(len(set(angle)))
#print(len(size))
#print(len(set(size)))

# -------------
# 유클리디안 거리를 이용해서 벡터간의 거리를 구하는 것은 비교할 대상이 있어야 하므로, 애매하다? 자기자신과??
# 여기서 가장 중요한 것은 디스크립터 벡터들끼리 얼마나 유사한지에 대해 알고싶은 것
# 유사한 디스크립터 벡터들이 많으면, 반복패턴을 가지는 이미지일 확률이 매우 높고, 매칭 성능이 떨어지게 된다.
# ... 그렇다면 기준 디스크립터 벡터를 만들어놓고, 거리를 구해, 자기자신의 크기???를 구하고 이를 리스트로 만들고 plot로 그려보면 되지 않을까? X X X 512라는 같은 길이가 나옴
# -------------
# 벡터 자체가 유사한지 확인해야함.
# Mean Shift Clustering 알고리즘을 사용하여 표본의 평균치가 변화하지 않을때 까지 반복하여 수치를 구한다?
# 이 결과 수치를 비교해서, 얼마나 유사한 디스크립터 벡터들이 많은지 확인할 필요가 있다.
# -------------
# matching을 할때에는 두 디스트립터 벡터간의 유사도를
