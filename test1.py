import numpy.matlib
import urllib.request
from PIL import Image
import numpy as np
from patchify import patchify
import matplotlib.pyplot as plt
import cv2
import matplotlib.pyplot as plt
from cart2polar import cart2polar
from getbin import getbin
from sklearn.preprocessing import MinMaxScaler
from scipy.io import savemat
import math



image1 = cv2.imread('1.jpg')
image2 = cv2.imread("2.jpg")

imgRgb1=cv2.resize(image1, (0, 0), fx = 1/5, fy = 1/5)
imgRgb2 = cv2.resize(image2, (0, 0), fx = 1/5, fy = 1/5)
# imgRgb1=cv2.resize(image1, (80, 80))
# imgRgb2=cv2.resize(image2, (80, 80))

print(imgRgb1.shape,'img1', imgRgb2.shape,'img2')
# cv2.imshow("lalala", imgRgb1) 
# cv2.waitKey(0)


def call_ssd(patch, region, alpha, center_patch):
    # patch_size=patch_size.shape
    region_size = region.shape

    SSD_region = np.zeros((region_size[0], region_size[1]))

    for row in range(center_patch[0]+1, region_size[1]-center_patch[1]):
        for col in range(center_patch[1], region_size[1]-center_patch[1]):
            temp = region[row-center_patch[0]:center_patch[0] +
                          row, col-center_patch[1]:center_patch[1]+col, :]

            diff = temp-patch
            SSD_region[row][col] = np.sum(diff**2)
            SSD_region[row][col] = np.exp(-alpha*SSD_region[row][col])

    return SSD_region


def get_self_sim_vec(SSD_region, bin, vec_size):
    self_similarities_vec = np.zeros((vec_size))
    
    
  

    num = 0
    for m in range(20):
        for n in range(4):
            temp = bin[m][n]
            
            max_value = 0
            temp_size = len(temp)

            for loc in range(temp_size):
                row, col = temp[loc]

                max_value = max(max_value, SSD_region[row][col])

            self_similarities_vec[num] = max_value
            num = num+1
    return self_similarities_vec

def com_Self_Similarities(imgRgb, region_size, patch_size, bin):
    lab_image = cv2.cvtColor(imgRgb, cv2.COLOR_BGR2LAB)

    lab_size = lab_image.shape
    vec_size = 80
    alpha = 1/(85**2)
    self_similarities = np.zeros(
        (lab_image.shape[0], lab_image.shape[1], vec_size))

    center_region = [math.floor(region_size[0]/2),
                     math.floor(region_size[1]/2)]
    center_patch = [math.floor(patch_size[0]/2), math.floor(patch_size[1]/2)]
    count = center_region[0]+1, lab_size[0] - center_region[0]
    for row in range(center_region[0]+1, lab_size[0] - center_region[0]):
        # print(f" {row}>{count}")
        for col in range(center_region[1]+1, lab_size[1]-center_region[1]):
            patch = lab_image[row-center_patch[0]:row + center_patch[0],
                              col-center_patch[0]:col + center_patch[0], :]
            region = lab_image[row - center_region[0]:row + center_region[0] + 1,
                               col - center_region[1]:col + center_region[1] + 1, :]
            SSD_region = call_ssd(patch, region, alpha, center_patch)
            vec = get_self_sim_vec(SSD_region, bin, vec_size)
            # print(vec.shape)
            vec = vec.reshape(-1, 1)

            scaler = MinMaxScaler(feature_range=(0, 1))
            scaler.fit_transform(vec)
            vec = vec.flatten()
            # print(vec)

            self_similarities[row, col, :] = vec
            # print(self_similarities.shape)
            # self_similarities[r]
            # print(a)

    return self_similarities


region_size = [46, 37]
patch_size = [5, 5]

radius, angle = cart2polar(region_size)
# print(radius)

bin = getbin(radius, angle, region_size)




self_similarities1 = com_Self_Similarities(imgRgb1, region_size, patch_size, bin)
# print(self_similarities1)

self_similarities2 = com_Self_Similarities(imgRgb2, region_size, patch_size, bin)
# print(self_similarities2)
  
  
  

width = 1
height = 1
center_sub = [math.floor(width/2), math.floor(height/2)]
p = 1
img_size1 = self_similarities1.shape
# print(center_sub)
# print(imgRgb1.shape)
# print(self_similarities1.shape,"s")
max_match = {}
match_score={}
sig_score_img = np.zeros((img_size1[0], img_size1[1]))
for row1 in range(center_sub[0], img_size1[0]-center_sub[0]):
    for col1 in range(center_sub[1], img_size1[1]-center_sub[1]):
        # print(row1,col1)
        sub1 = self_similarities1[row1 - center_sub[0]:row1 +center_sub[0]+1, col1 - center_sub[1]:col1 + center_sub[1]+1, :]
        temp1 = np.tile(sub1, (self_similarities2.shape[0], self_similarities2.shape[1], 1))
        temp2 = -1 * np.sum((self_similarities2 - temp1) ** 2, axis=2)
        max_match[1] = np.max(np.max(temp2))
        # print(max_match)
        match_score[1] = (temp2)
        # print(match_score)
        # avgMatch=0
        # stdMatch=0
        # sum_max_match=0
        # avgMatch+=np.mean(temp2)
        # stdMatch+=np.std(temp2)
        # sum_max_match+=max_match[1]
        # sig_score_img[row1][col1] = (sum_max_match-avgMatch)/stdMatch
        # print(sig_score_img)
        # print(row1,col1)

exit()
ma =np.max(np.max(sig_score_img))
mi=np.min(np.min(sig_score_img))

arr=sig_score_img
max_value=0
positions = []
for i in range(arr.shape[0]):  # Loop through rows
        for j in range(arr.shape[1]):  # Loop through columns
            if arr[i, j] == max_value:
                positions.append((i, j))

print(positions)
norm_sig_score_img = (sig_score_img - mi)/(ma-mi)
norm_sig_score_img = norm_sig_score_img*255



# print(norm_sig_score_img.shape)
# x = find_element_2d(sig_score_img, ma)
x, y = np.where(sig_score_img == ma)
print(x)