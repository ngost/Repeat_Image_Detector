from proto_repeat_detector import classify_v1,classify_v2, classify_v3, classify_v4, classify_v5,classify_test_jong
import scipy as sp
import csv
import time


def list_classify(imgs):
    i = 0
    for img in imgs:
        i += 1
        result = classify_v1(img)
        print(str(i)+"/"+str(len(imgs)))
        if result is 0:
            continue

        if result < 0.5:
            print("repeat_img / "+str(result))
            print(img)
        else:
            print("general_img / "+str(result))



def list_classify2(imgs):
    i = 0
    for img in imgs:
        i += 1
        result = classify_v2(img)
        print(str(i)+"/"+str(len(imgs)))
        if result is 0:
            continue

        if result > 0.7:
            print("repeat_img / "+str(result))
            print(img)
        else:
            print("general_img / "+str(result))


def list_classify3(imgs):
    i = 0
    for img in imgs:
        i += 1
        result = classify_v3(img)
        print(str(i)+"/"+str(len(imgs)))
        if result is 0:
            continue

        if result > 0.04:
            print("repeat_img / "+str(result))
            print(img)
        else:
            print("general_img / "+str(result))


def list_classify4(imgs):
    i = 0
    for img in imgs:
        result = classify_v5(img)
        print(str(i)+"/"+str(len(imgs)))
        i += 1
        if result is 0:
            continue

        if result < 0.1:
            print("repeat_img / "+str(result))
            print(img)
            print("")
        else:
            print("general_img / "+str(result))
            print(img)
            print("")


def list_classify_test(imgs):
    i = 0
    for img in imgs:
        i += 1
        result = classify_test_jong(img)
        print(str(i)+"/"+str(len(imgs)))
        if result is 0:
            continue

        if result > 0.6:
            print("repeat_img / "+str(result))
            print(img)
        else:
            print("general_img / "+str(result))
            print(img)



def read_img():
    img_urls = []
    f = open('open-images.tsv', 'r', encoding='utf-8')
    rdr = csv.reader(f, delimiter='\t')
    r = list(rdr)
    for r_ele in r:
        if int(str(r_ele[1])) < 50000:
            img_urls.append(str(r_ele[0]))

    return img_urls

#print(img_urls)
start_time = time.time()
repeat_imgs = ['img/r1.jpg','img/r2.jpg','img/r3.jpg','img/r7.jpg','img/r8.jpg','img/r9.png','img/r10.jpeg','img/r11.jpeg','img/r12.png','img/r13.png','img/r15.jpg','img/r16.jpg','img/r17.jpg','img/r18.png','img/r19.jpg','img/r20.jpg']
general_imgs = ['img/g1.jpeg','img/low8.jpeg','img/low7.jpeg','img/low6.jpeg','img/low5.jpeg','img/g6.png','img/g7.jpeg','img/g8.jpeg','img/g9.jpeg','img/low4.jpeg','img/g11.jpeg','img/g12.jpeg']
#url_imgs =['https://c7.staticflickr.com/6/5499/10245691204_98dce75b5a_o.jpg']

#list_classify(repeat_imgs)
#list_classify2(repeat_imgs)
#list_classify3(repeat_imgs)
#list_classify4(repeat_imgs)
#list_classify_test(repeat_imgs)
#list_classify(general_imgs)
#list_classify2(general_imgs)
#list_classify3(general_imgs)
#list_classify4(general_imgs)
#list_classify(img_urls)
print("\n")
#list_classify_test(general_imgs)
list_classify_test(['img/r21.jpg','img/r22.jpg','img/r23.jpg','img/r24.jpg'])
#list_classify(['img/r11.jpeg'])
#list_classify2(['img/r11.jpeg'])

#img_url = read_img()
#list_classify(img_url)
#list_classify2(img_url)
#list_classify_test(img_url)
print("--- %s seconds ---" % (time.time() - start_time))