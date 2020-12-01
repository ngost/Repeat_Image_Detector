import math
def get_entropy(nphist):

    pdf_list = nphist

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


print(get_entropy([0.016666666,0.016666666,0.016666666,0.016666666,0.016666666,0.016666666,0.016666666,0.016666666,0.016666666,0.016666666,0.016666666,0.016666666,0.016666666,0.016666666,0.016666666,0.016666666,0.016666666,0.016666666,0.016666666,0.016666666,0.016666666,0.016666666,0.016666666,0.016666666,0.016666666,0.016666666,0.016666666,0.016666666,0.016666666,0.016666666,0.016666666,0.016666666,0.016666666,0.016666666,0.016666666,0.016666666,0.016666666,0.016666666,0.016666666,0.016666666,0.016666666,0.016666666,0.016666666,0.016666666,0.016666666,0.016666666,0.016666666,0.016666666,0.016666666,0.016666666,0.016666666,0.016666666,0.016666666,0.016666666]))