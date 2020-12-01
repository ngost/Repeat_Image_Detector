from vector_similarity import euclidean,different
from skimage import io
import cv2
from scipy.stats import norm
import scipy.stats as stats
import numpy as np
from matplotlib import pyplot as plt
import math

def classify_v5(img_name):
    try:

        img = cv2.imread(img_name)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.blur(gray, (3, 3))
        sift = cv2.xfeatures2d.SIFT_create()
        kp, des = sift.detectAndCompute(gray, None)
        # img = cv2.drawKeypoints(gray, kp, img,)

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

            sum_data = sum_data / kp_num
            des_sum.append(sum_data)
            # print(sum_data)
            sum_data = 0
            i += 1

        # print(des_sum) #des_sum = descriptor들의 평균
        # print(des[0])

        euclidean_result = []
        for descriptor in des:
            euclidean_result.append(round(euclidean(des_sum, descriptor)))

      #  print(euclidean_result)
        bins_num = max(euclidean_result)-min(euclidean_result)
     #   print(bins_num)
        # bins_num = 30
        # bins_num = max(euclidean_result)-min(euclidean_result)

        euclidean_result = norm_list(euclidean_result, max(euclidean_result), min(euclidean_result))
        # norm_euc = euclidean_result
        # euclidean_result.sort()
       # print(euclidean_result)

        count, bins, ignored = plt.hist(euclidean_result, bins_num, color='red', density=True)
        # print("bin")
        # print(count)
        # print(len(count))
        # sample들을 이용해서 Gaussian Distribution의 shape을 재구축해서 line으로 그린다.
        # plt.show()
        sigma = np.std(euclidean_result)
        mu = np.mean(euclidean_result)

        # sigma = np.std(list(set(count)))
        # mu = np.mean(list(set(count)))

      #  print(sigma)
       # print(mu)

        plt.plot(bins, 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(- (bins - mu) ** 2 / (2 * sigma ** 2)), linewidth=1,
                 color='b')

        # plt.hist(euclidean_result, facecolor='red',bins=50)  # arguments are passed to np.histogram
        plt.title(img_name)
#        plt.show()
        # avg_list(norm_e)
        # des1 = des[]
        # des2 = des[2]
        # print(des1)
        # print(des2)

        # print(len(set(euclidean_result)))
        # result = 1/(sigma * np.sqrt(2 * np.pi)) * np.exp(-(bins - mu)**2 / (2 * sigma**2))
        test_index = []

        for i in range(bins_num):
            test_index.append(i / bins_num)
        # print(test_index)
        test = 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-(bins - mu) ** 2 / (2 * sigma ** 2))
        # test
        # print("test")

        # test : 실제 euclidean_data에서 값이 존재하는 index들에 대해, 연속정규분포 함수에 식을 대입하여 얻은 정규분포 데이터 (정규분포에서 나온 비교대상)
        # hist_pdf : index 값에 따른 histgram의 pdf 데이터들
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

        # print("에러율 : "+str((sum_error/bin_num)))
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

        # print("범위 :")
        # print(len(count))
        # print(len(test))

        # print("에러율2:")
        # count.sort()
        # test.sort()
        # count는
  #      print(count)
        i = 0
        del_list = []
        for count_e in count:
            if count_e == 0.0:
                del_list.append(i)
            i += 1

 #       print(del_list)
        # print(count[2])
        count = count.tolist()
#        print(count)
        test = test.tolist()

        for del_e in reversed(del_list):
            count.pop(del_e)
            test.pop(del_e)

        count.sort()
        test.sort()
        real_result = euclidean(count, test)
        # for count_e,test_e in zip(count,test):
        #     val = (count_e - test_e) ** 2
        #     if val>1:
        #         print(val)
        #         print(count_e)
        #         print(test_e)
        #         print("")
        # print(1/(1+real_result))
        i = 0

        sum_error = 0

        return 1 / (1 + real_result)
    except cv2.error as e:
        print("Oops!")
        return 0
    except TypeError as e:
        print("Type Oops!")
        return 0
    except ZeroDivisionError as e:
        print("norm_error!")
        return 0



def avg_list(lst):
    lst_sum = sum(lst)
    return lst_sum/len(lst)


def norm_list(lst,lengh_max,lengh_min):
    # 정규화
    maxim = lengh_max
    minim = lengh_min

    norm_data = []
    for lst_ele in lst:
        norm_data.append((lst_ele - minim) / (maxim - minim))

    return norm_data


def classify_v1(img_name):

    try:
        img = cv2.imread(img_name)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
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

            sum_data = sum_data / kp_num
            des_avg.append(round(sum_data))
            sum_data = 0
            i += 1

        euclidean_result = []

        for descriptor in des:
            euclidean_result.append(round(euclidean(des_avg, descriptor)))

        # 정규화
        norm_data = norm_list(euclidean_result,max(euclidean_result),min(euclidean_result))

        # 정규분포 그리기

        mean = avg_list(norm_data)  # 평균
        std = np.std(norm_data, ddof=1)  # 표준편차
        data = np.random.normal(mean, std, len(norm_data))

        norm_data.sort()
        data.sort()
        result_final = euclidean(norm_data, data)
        return 1/(1+result_final)

    except cv2.error as e:
        print("Oops!")
        return 0
    # except TypeError as e:
    #     print("Type Oops!")
    #     return 0
    except ZeroDivisionError as e:
        print("norm_error!")
        return 0


def classify_v2(img_name):
    try:
        bin_num = 100
        img = io.imread(img_name)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.blur(gray, (3, 3))
        sift = cv2.xfeatures2d.SIFT_create()
        kp, des = sift.detectAndCompute(gray, None)

        i = 0
        sum_data = 0
        kp_num = len(kp)
        des_sum = []
        while i < 128:
            for descriptor in des:
                sum_data = sum_data + descriptor[i]

            sum_data = sum_data / kp_num
            des_sum.append(round(sum_data))
            sum_data = 0
            i += 1

        euclidean_result = []
        for descriptor in des:
            euclidean_result.append(round(euclidean(des_sum, descriptor)))

        norm_euc = norm_list(euclidean_result,max(euclidean_result),min(euclidean_result))

        count, bins, ignored = plt.hist(norm_euc, bin_num, normed=True)

        sigma = np.std(norm_euc)
        mu = np.mean(norm_euc)

        result = 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-(bins - mu) ** 2 / (2 * sigma ** 2))

        i = 0
        sum_error = 0
        while i < bin_num:
            sum_error += (((result[i] + result[i + 1]) / 2) - count[i]) ** 2
            i += 1

        return sum_error/bin_num
    except cv2.error as e:
        print("Oops!")
        return 0
    except TypeError as e:
        print("Type Oops!")
        return 0
    except ZeroDivisionError as e:
        print("norm_error!")
        return 0


def classify_v3(img_name):
    try:
        img = io.imread(img_name)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.blur(gray, (3, 3))
        sift = cv2.xfeatures2d.SIFT_create()
        kp, des = sift.detectAndCompute(gray, None)

        height, width, channels = img.shape

        i = 0
        sum_data = 0
        kp_num = len(kp)
        des_sum = []
        while i < 128:
            for descriptor in des:
                sum_data = sum_data + descriptor[i]

            sum_data = sum_data / kp_num
            des_sum.append(round(sum_data))
            # print(sum_data)
            sum_data = 0
            i += 1

        euclidean_result = []
        for descriptor in des:
            euclidean_result.append(round(euclidean(des_sum, descriptor)))

        count, bins, ignored = plt.hist(euclidean_result, max(euclidean_result) - min(euclidean_result), normed=True,
                                        color='red')
        sigma = np.std(euclidean_result)
        mu = np.mean(euclidean_result)

        hist = np.histogram(euclidean_result, bins=max(euclidean_result) - min(euclidean_result))
        hist = hist[0]
        sum_hist = sum(hist)

        hist_pdf = []
        for hist_e in hist:
            hist_pdf.append(hist_e / sum_hist)

        test_index = []

        for i in range(max(euclidean_result) - min(euclidean_result)):
            test_index.append(i + min(euclidean_result))

        test = 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-(test_index - mu) ** 2 / (2 * sigma ** 2))
        # test
        i = 0

        sum_error = 0
        count.sort()
        test.sort()
        print(count)
        print(test)
        # while i < len(hist_pdf):
        #     sum_error += ((hist_pdf[i] - test[i]) ** 2)
        #     i += 1
        result_final = euclidean(count,test)
        return 1/(1+result_final)
 #       result2 = (sum_error / len(count)) * 10000

#        return result2
    except cv2.error as e:
        print("Oops!")
        return 0
    except TypeError as e:
        print("Type Oops!")
        return 0
    except ZeroDivisionError as e:
        print("norm_error!")
        return 0


def classify_v4(img_name):
    try:
#        img_name = 'img/g7.jpeg'
        img = io.imread(img_name)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.blur(gray, (3, 3))
        sift = cv2.xfeatures2d.SIFT_create()
        kp, des = sift.detectAndCompute(gray, None)
        # img = cv2.drawKeypoints(gray, kp, img,)

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

            sum_data = sum_data / kp_num
            des_sum.append(round(sum_data))
            # print(sum_data)
            sum_data = 0
            i += 1

        # print(des_sum) #des_sum = descriptor들의 평균
        # print(des[0])

        euclidean_result = []
        for descriptor in des:
            euclidean_result.append(round(euclidean(des_sum, descriptor)))

        # print(euclidean_result)
        bins_num = len(set(euclidean_result))
        euclidean_result = norm_list(euclidean_result, max(euclidean_result), min(euclidean_result))
        # norm_euc = euclidean_result

        count, bins, ignored = plt.hist(euclidean_result, bins_num, density=True, color='red')
        # print("bin")
        # print(count)
        # print(len(count))
        # sample들을 이용해서 Gaussian Distribution의 shape을 재구축해서 line으로 그린다.
        # plt.show()
        sigma = np.std(euclidean_result)
        mu = np.mean(euclidean_result)
        # print(sigma)
        # print(mu)
        plt.plot(bins, 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(- (bins - mu) ** 2 / (2 * sigma ** 2)), linewidth=2,
                 color='b')

        # plt.hist(euclidean_result, facecolor='red',bins=50)  # arguments are passed to np.histogram
        plt.title(img_name)
#        plt.show()
        # avg_list(norm_e)
        # des1 = des[]
        # des2 = des[2]
        # print(des1)
        # print(des2)

        # print(len(set(euclidean_result)))
        result = 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-(bins - mu) ** 2 / (2 * sigma ** 2))
        test_index = []

        for i in range(bins_num):
            test_index.append(i / bins_num)
        # print(test_index)
        #test = 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-(test_index - mu) ** 2 / (2 * sigma ** 2))
        test = 1/(sigma * np.sqrt(2 * np.pi)) * np.exp(-(bins- mu)**2 / (2 * sigma**2))
        # test
        # print("test")

        # test : 실제 euclidean_data에서 값이 존재하는 index들에 대해, 연속정규분포 함수에 식을 대입하여 얻은 정규분포 데이터 (정규분포에서 나온 비교대상)
        # hist_pdf : index 값에 따른 histgram의 pdf 데이터들
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

        # print("에러율 : "+str((sum_error/bin_num)))
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

        # print("범위 :")
        # print(len(count))
        # print(len(test))

        # print("에러율2:")
        count.sort()
        test.sort()
        # print(count)
        # print(test)
        real_result = euclidean(count, test)
        # print(1/(1+real_result))
        i = 0

        sum_error = 0

        return 1/(1+real_result)
    except cv2.error as e:
        print("Oops!")
        return 0
    except TypeError as e:
        print("Type Oops!")
        return 0
    except ZeroDivisionError as e:
        print("norm_error!")
        return 0


def classify_test_jong(img_name):
    try:
        euclidean_result = []
        sum_data = 0
        des_sum = []
        kp_num = 0

        img = io.imread(img_name)
        w, h = img.shape[:2]

        if w > 1000:
            img = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        try:
            w, h, c = img.shape
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        except ValueError as e:

            gray = img

        gray = cv2.blur(gray, (3, 3))
        sift = cv2.xfeatures2d.SIFT_create()
        kp, des = sift.detectAndCompute(gray, None)

        kp_num = len(kp)

        i = 0
        while i < 128:
            for descriptor in des:
                sum_data = sum_data + descriptor[i]

            sum_data = sum_data / kp_num
            des_sum.append(sum_data)
            sum_data = 0
            i += 1


        for descriptor in des:
            euclidean_result.append(euclidean(des_sum, descriptor))

        bins_num = len(set(euclidean_result))


        euclidean_result = norm_list(euclidean_result, max(euclidean_result), min(euclidean_result))
        count, bins, ignored = plt.hist(euclidean_result, bins_num, color='red', density=True)

        sigma = np.std(euclidean_result)
        mu = np.mean(euclidean_result)


        bins2 = []
        for bins_e in bins:
            bins2.append(-bins_e)
        bins2.sort(reverse=False)

        bins2.extend(bins)

        test = 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-(bins2 - mu) ** 2 / (2 * sigma ** 2))

        count = count.tolist()
        count2 = []
        for count2_e in range(len(count)):
            count2.append(0)

        count2.extend(count)
        test = test.tolist()

        real_result2 = different(count2, test)
        final_result = real_result2 / len(test)

        test = []
        count2 = []
        count = []

        return final_result

    except cv2.error as e:
        print("Oops!")
        return 0
    except TypeError as e:
        print("Type Oops!")
        return 0
    except ZeroDivisionError as e:
        print("norm_error!")
        return 0
