import math as math
import cmath as cmath


def calcDistance(post_lat, post_long, lat, long):

    distance = (6371*cmath.acos(cmath.cos(math.radians(lat))
               * cmath.cos(math.radians(post_lat))
               * cmath.cos(math.radians(post_long)-math.radians(long))
               + cmath.sin(math.radians(long))
               * cmath.sin(math.radians(post_lat))))
    return distance


def calcDistance2(post_lat, post_long, lat, long):
    data = (6371 * cmath.acos(
        cmath.cos(math.radians(lat)) * cmath.cos(math.radians(post_lat)) * cmath.cos(math.radians(post_long)
                                                                                            - math.radians(
            long)) + cmath.sin(math.radians(lat)) * cmath.sin(math.radians(post_lat))))

    result = float(data.real)
    return result
#
# print(calcDistance(127.123421,36.423124,127.123421,36.423124))
# data = calcDistance2(36.423124,127.123421,37.423124,127.123421)
# print(data)
#
# print(type(data))
# real = round(float(data.real),2)
#
# print(real)
# if float(data.real) > 9:
#     print(float(data.real))
# else:
#     print(float(data.real))


import time
def readTest():
    start_time = time.time()

    from skimage import io
    io.imread("https://t1.daumcdn.net/cfile/tistory/2608F74458D838384E")
    print("--- %s seconds ---" % (time.time() - start_time))

readTest()



