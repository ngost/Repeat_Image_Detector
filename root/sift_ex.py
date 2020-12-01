import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from skimage import io
import time

def resizing_img(img_para):
    height, width = img_para.shape[:2]
    thumbnail = cv.resize(img_para, (round(width/5), round(height/5)), interpolation = cv.INTER_AREA)
    return thumbnail


def matching_sift(query_img, train_img):
    MIN_MATCH_COUNT = 10
    img1 = query_img       # queryImage
    img2 = io.imread(train_img) # trainImage
    img1 = resizing_img(img1)
    img2 = resizing_img(img2)

    try:
        w, h, c = img2.shape
        gray1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
        gray2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)
    except ValueError as e :

        gray1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
        gray2 = img2


    # Initiate SIFT detector
    sift = cv.xfeatures2d.SIFT_create()
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(gray1,None)
    kp2, des2 = sift.detectAndCompute(gray2,None)
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)
    flann = cv.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1,des2,k=2)
    # store all the good matches as per Lowe's ratio test.
    good = []
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            good.append(m)

        if len(good) > MIN_MATCH_COUNT:
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
            M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
            matchesMask = mask.ravel().tolist()
            h, w, d = img1.shape
            pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
            dst = cv.perspectiveTransform(pts, M)
            img2 = cv.polylines(img2, [np.int32(dst)], True, 255, 3, cv.LINE_AA)
        else:
           print("Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT))
            matchesMask = None

    if len(good) > 50:
        draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                           singlePointColor = None,
                           matchesMask = matchesMask, # draw only inliers
                           flags = 2)
        img3 = cv.drawMatches(img1, kp1, img2, kp2, good, None, **draw_params)
        return len(good)
    else:
    return len(good)