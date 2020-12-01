from vector_similarity import euclidean,different
from skimage import io
import cv2
from scipy.stats import norm
import scipy.stats as stats
import numpy as np
from matplotlib import pyplot as plt
import math
import csv


def read_img():
    img_urls = []
    f = open('open-images.tsv', 'r', encoding='utf-8')
    rdr = csv.reader(f, delimiter='\t')
    r = list(rdr)
    for r_ele in r:
        if int(str(r_ele[1])) < 500000:
            img_urls.append(str(r_ele[0]))

    return img_urls


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

def norm_z(lst,avg,std):
    norm_data = []
    for lst_ele in lst:
        norm_data.append((lst_ele - avg) / std)

    return norm_data

def classify_test_jong(img_name):
    try:
        euclidean_result = []
        sum_data = 0
        des_sum = []
        kp_num = 0


        #step1 이미지 로드
        img = io.imread(img_name)
        w, h = img.shape[:2]


        #step2 이미지 사이즈 조정
        if w > 1000:
            img = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        try:
            w, h, c = img.shape
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        except ValueError as e:

            gray = img

        #step3. sift Detector로 keypoint, descriptor vector 추출
        gray = cv2.blur(gray, (3, 3))
        sift = cv2.xfeatures2d.SIFT_create()
        kp, des = sift.detectAndCompute(gray, None)

        kp_num = len(kp)
        kp_e_list = []
        num_3 = 0
        for kp_e in kp:
            #print(kp_e.size)
            kp_e_list.append(kp_e.size)
            if kp_e.size <2:
                num_3 +=1
#            print(kp_e.size)
        print(num_3)
        print(num_3/len(kp))
        plt.hist(kp_e_list,bins=len(kp))
        plt.show()

        #step4. descritor의 각각 차원마다의 평균 값을 가지는 평균 descriptor 계산
        i = 0
        while i < 128:
            for descriptor in des:
                sum_data = sum_data + descriptor[i]

            sum_data = sum_data / kp_num
            des_sum.append(sum_data)
            sum_data = 0
            i += 1


        #step5. 각각의 keypoint의 descriptor vector와 평균 descripto 간의 유클리디언 거리 계산( 차원수를 1차원으로 감소)
        for descriptor in des:
            euclidean_result.append(euclidean(des_sum, descriptor))



        #step6. histogram의 bins 개수 결정 (존재하는 데이터 범위만큼)
        bins_num = len(euclidean_result)


        #step7. 1차원으로 감소된 각각의 keypoint들의 descriptor와 avg descriptor 사이의 유클리디언 거리값 -> 리스트를 평균이 0, 표준편차가 1인 z-분포로 정규화
#        euclidean_result = norm_list(euclidean_result, max(euclidean_result), min(euclidean_result))
        sigma = np.std(euclidean_result)
        mu = np.mean(euclidean_result)

        euclidean_result = norm_z(euclidean_result,mu,sigma)

        #step8. bin을 kp개수만큼 가지는 히스토그램 draw, 이때, count값은 density를 이용하여 확률값으로 변환?
        count, bins, ignored = plt.hist(euclidean_result, bins=kp_num, color='red',density=True)


        #step9.  정규화된 데이터의 표준편차와 평균을 구함
        sigma = np.std(euclidean_result)
        mu = np.mean(euclidean_result)


        # #                                                             효과 미미step10. histogram의 bins를 0~1 에서 -1 ~ 1 까지로 강제로 늘림
        # # 실제로 data는 0 이하의 영역에서 존재하지 않음
        #
        # bins2 = []
        # for bins_e in bins:
        #     bins2.append(-bins_e)
        # bins2.sort(reverse=False)
        # bins2.extend(bins)

#        plt.show()
        #데이터의 평균과 표준편차를 이용하여

        #step10. data의 평균과 표준편차로 표준 정규분포 N(0,1)를 그림  ->그릴때 사용되는 데이터-> hist의 count값
        plt.plot(bins, 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(- (bins - mu) ** 2 / (2 * (sigma ** 2))), linewidth=3,
                 color='b')



        #step.11 정규분포 수식에 들어갈 데이터 생성
        bins_new = []
        i = 0
        for bin_e in bins:
            if i<len(bins)-1:
                bins_new.append((bins[i]+bins[i+1])/2)
                i += 1
            else:
                break

        #step.12 수식에 적용하여 bins 범위에 따른 정규분포 y축 값 리스트를 얻음
        dist_count = 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-(bins_new - mu) ** 2 / (2 * sigma ** 2))
        # count = count.tolist()
        # count2 = []
        # for count2_e in range(len(count)):
        #     count2.append(0)
        #
        # count2.extend(count)

        #step. 13 비교
        #차이를 모두 더한 후, 개수로 나눈 값 (에러의 평균)

        error_avg = different(count, dist_count)
        #차이를 모두 더한 후, 루트 (에러의 누적?)
        error_cumu = euclidean(count,dist_count)
        print(len(dist_count))
        print(error_avg)
        error_avg = error_avg / len(dist_count)
        error_cumu = 1/(1+error_cumu)

#        plt.title(img_name+"\n"+str(error_avg)+"\n"+str(error_cumu))
        plt.show()
        return error_avg,error_cumu

    except cv2.error as e:
        print("Oops!")
        return 0,0
    except TypeError as e:
        print("Type Oops!")
        return 0,0
    except ZeroDivisionError as e:
        print("norm_error!")
        return 0,0


def list_classify_test(imgs):
    i = 0
    for img in imgs:
        i += 1
        result1, result2 = classify_test_jong(img)
        print(str(i)+"/"+str(len(imgs)))
        if result1 is 0:
            continue

        if result1 > 0.0443:
            print("repeat_img / "+str(result1)+str(result2))
            print(img)
        else:
            if result2 < 0.1:
                print("repeat_img r2 / " + str(result1)+str(result2))
                print(img)
            else:
                print("general_img")
                print(result1)
                print(result2)
                print(img)


repeat_imgs = ['img/r1.jpg','img/r2.jpg','img/r3.jpg','img/r5.png','img/r7.jpg','img/r8.jpg','img/r9.png','img/r10.jpeg','img/r11.jpeg','img/r12.png','img/r13.png','img/r14.jpg','img/r15.jpg','img/r16.jpg','img/r17.jpg','img/r18.png','img/r19.jpg','img/r20.jpg']
general_imgs = ['img/g1.jpeg','img/low8.jpeg','img/low7.jpeg','img/low6.jpeg','img/low5.jpeg','img/g6.png','img/g7.jpeg','img/g8.jpeg','img/g9.jpeg','img/low4.jpeg','img/g11.jpeg','img/g12.jpeg']
#img_url = read_img()
#list_classify_test(['img/r19.jpg'])
#list_classify_test(img_url)
list_classify_test(['img/g1.jpeg'])
#list_classify_test(repeat_imgs)
#list_classify_test(general_imgs)
#list_classify_test(['img/center-feature.jpeg'])