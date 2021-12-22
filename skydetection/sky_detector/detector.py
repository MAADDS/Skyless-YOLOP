import cv2
from statistics import median
from scipy.signal import medfilt
from scipy import ndimage
import numpy as np
from matplotlib import pyplot as plt


def cal_skyline(mask):
    h, w = mask.shape
    greatest = 0
    zerolist = []
    zerolist2 = []
    print(w)    
    for i in range(w):
        raw = mask[:, i]
        after_median = medfilt(raw, 19)
        try:
            
            first_zero_index = np.where(after_median == 0)[0][0]
            first_one_index = np.where(after_median == 1)[0][0]
            if greatest < first_zero_index :
                greatest = first_zero_index
            if first_zero_index != 0:
                zerolist2.append(first_zero_index)
            zerolist.append(first_zero_index)
            
            #if first_zero_index > 20:
            mask[first_one_index:first_zero_index, i] = 0
            mask[first_zero_index:, i] = 1
            mask[:first_one_index, i] = 1
        except:
            continue
    medline = int(median(zerolist2))
    meanline = int(np.mean(zerolist2))

    return mask, medline, meanline, greatest


def get_sky_region_gradient(img):

    h, w, _ = img.shape
    print(w)

    blurshape = (5,5)

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    plt.imshow(img_gray)
    plt.show()

    img_gray = cv2.blur(img_gray, blurshape)
    plt.imshow(img_gray)
    plt.show()
    cv2.medianBlur(img_gray, 15)
    lap = cv2.Laplacian(img_gray, cv2.CV_8U)
    gradient_mask = (lap < 6).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, blurshape)
    mask = cv2.morphologyEx(gradient_mask, cv2.MORPH_ERODE, kernel)
    #plt.imshow(mask)
    #plt.show()

    mask, medline, meanline, greatest = cal_skyline(mask)
    #plt.imshow(mask)
    #plt.show()
    temp = 130
    after_img = cv2.bitwise_and(img, img, mask=mask)
    after_img[meanline-5:meanline+5,:] = (255,0,255)
    #after_img[int(np.mean([medline,temp]))-5 : int(np.mean([medline,temp]))+5,:] = (255,255,0)
    #after_img[temp-5:temp+5,:] = (0,255,0)

    return after_img