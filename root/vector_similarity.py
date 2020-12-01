from matplotlib import pyplot as plt
import numpy as np
import cv2
import math
from scipy.stats import norm


def euclidean(vec1, vec2):
    result = sum([(xi - yi) ** 2 for xi, yi in zip(vec1, vec2)])
    return math.sqrt(result)


def different(vec1, vec2):
    diff_list = [(xi - yi) ** 2 for xi, yi in zip(vec1, vec2)]
#    print(max(diff_list))
    sort_list = diff_list.copy()
    sort_list.sort(reverse=True)
    indexs = []
    print(diff_list)
    for sort in sort_list:
        print(sort)
        if sort<1.0:
            indexs = None
            break
        indexs.append(diff_list.index(sort))
        if len(indexs)>1:
            break


    print(indexs)

    return sum(diff_list),max(diff_list)

#
# img = cv2.imread('img/2.jpg')
# gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# sift = cv2.xfeatures2d.SIFT_create()
# kp, des = sift.detectAndCompute(gray, None)
# #img = cv2.drawKeypoints(gray, kp, img,)
#
#
#
# positionX = []
# positionY = []
# # for point in kp:
# #     temp = point.size
# #     #temp = point.pt[0]
# #     print(temp)
# #     positionX.append(temp)
# #     temp = point.angle
# #     #temp = point.pt[1]
# #     print(temp)
# #     positionY.append(temp)
# height, width, channels = img.shape
#
# des1 = des[2]
# des2 = des[2]
# print(des1)
# print(des2)
# print( euclidean(des1,des2))
