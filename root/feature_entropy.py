from matplotlib import pyplot as plt
import numpy as np
import cv2
import math
from scipy.stats import norm

def get_entropy(pixel_size, nphist):
    hist = nphist[0].tolist()
    print(hist)
    pdf_list = []
    w_count = {}

    for lst in hist:
        try:
            w_count[lst] += 1
        except:
            w_count[lst] = 1
    #print(w_count)

    key_list = w_count.keys()
    print(key_list)

    for key in key_list:
        pdf_list.append(w_count[key]/(pixel_size-1))

    print(pdf_list)

    temp_pdf = 0
    for pdf_list_e in pdf_list:
        temp_pdf += pdf_list_e

    print(temp_pdf)
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

def get_entropy2(pixel_size, nphist):
    hist = nphist[0].tolist()
    feature_num = 0
    pdf_list = []

#get feature num
    pdf_zero =0

    for hist_element in hist:
        feature_num += hist_element

    for hist_e in hist:
        if hist_e is 0:
            pdf_zero += 1
        else:
            pdf_list.append(hist_e/pixel_size)

    pdf_list.append(pdf_zero/pixel_size)
    print("pdf list : ")
    print(pdf_list)

    sumresult = 0
    for pdf_list_e in pdf_list:
        sumresult += pdf_list_e

    print(sumresult)

    #real = sum(hist)
#    feature_number = real
    #hist = [1]
    #print(hist)
    entropy = 0
    percent = 0
    zero = 0
    for data in pdf_list:
        if data is not 0:
            entropy += data*math.log2(data)
        #percent = percent + (data/pixel_size)

#    entropy = entropy + (zero/pixel_size)*math.log2(zero/pixel_size)

    #print(pixel_size)
    #print(len(hist)+1)
    #print(pixel_size/(len(hist)+1))

    #print(sum(hist))

#    percent = percent + zero/pixel_size

    #print(percent)
    return -entropy


img = cv2.imread('img/r3.jpg')
gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
sift = cv2.xfeatures2d.SIFT_create()
kp = sift.detect(gray, None)
img = cv2.drawKeypoints(gray, kp, img,)

positionX = []
positionY = []

for point in kp:
    temp = int(round(point.pt[0]))
    #temp = point.pt[0]
    positionX.append(temp)
    temp = int(round(point.pt[1]))
    #temp = point.pt[1]
    positionY.append(temp)
height, width, channels = img.shape
print(height)
print(width)
plt.plot(positionX, positionY, 'ro', markersize=1)
plt.axis([0, width, height, 0])
plt.show()

#print(positionX)
#print(positionY)


#cv2.imwrite('wood.jpg',img)

plt.hist(positionX, facecolor='blue',bins = width, range=[0,width+1])  # arguments are passed to np.histogram
plt.title("Entropy X")
plt.show()

plt.hist(positionY, facecolor='red',bins = height,range=[0,height+1])  # arguments are passed to np.histogram
plt.title("Entropy Y")
plt.show()

#feature_num = len(set(positionX))
nphistx = np.histogram(positionX, bins=range(width))
#print(nphistx[0].tolist())
print('x entropy : ' + str(get_entropy(width, nphistx)))

#feature_num = len(set(positionY))
nphisty = np.histogram(positionY, bins=range(height))
#print(nphisty[0].tolist())



print('y entropy : ' + str(get_entropy(height, nphisty)))
