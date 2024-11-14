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
import sys
import random









def call_ssd(patch, region, alpha, center_patch):
    patch_size=patch.shape
   
    region_size = region.shape
 

    

    SSD_region = np.zeros((region_size[0], region_size[1]))

    for row in range(center_patch[0], region_size[0]-center_patch[0]):
        for col in range(center_patch[1], region_size[1]-center_patch[1]):
          
            temp = region[row-center_patch[0]:center_patch[0] +row+1, col-center_patch[1]:center_patch[1]+col+1, :]
            
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
                
                # print(row,col)

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

    for row in range(center_region[0], lab_size[0] - center_region[0]):
        
        for col in range(center_region[1], lab_size[1]-center_region[1]):
            
            print(f"row :{row} >> col :{col}")
            
            patch = lab_image[row-center_patch[0]:row + center_patch[0]+1,col-center_patch[0]:col + center_patch[0]+1, :]
        
            region = lab_image[row - center_region[0]:row + center_region[0]+1,col - center_region[1]:col + center_region[1]+1 , :]
         
         
          
            # print(f"patch{row-center_patch[0]}:{row + center_patch[0] } >> {col-center_patch[0]}:{col + center_patch[0]}")
            # print(f"{row - center_region[0]}:{row + center_region[0] } >> {col - center_region[1]}:{col + center_region[1] }")
            
            SSD_region = call_ssd(patch, region, alpha, center_patch)
      
         
            
         
            vec = get_self_sim_vec(SSD_region, bin, vec_size)
                   
      
            
            vec = vec.reshape(-1, 1)
           

            scaler = MinMaxScaler(feature_range=(0, 1))
            vec=(scaler.fit_transform(vec))
           
            
            # print(scaler.fit_transform(vec))
            # scaler.fit_transform(vec).flatten()
            vec = vec.flatten()
     
            self_similarities[row, col, :] = vec
          
         
 

    return self_similarities


def dr(image, sig_score_img, region_size, scale):
    
    ma =np.max(np.max(sig_score_img))
    mi=np.min(np.min(sig_score_img))
    norm_sig_score_img = (sig_score_img - mi)/(ma-mi)
    norm_sig_score_img = norm_sig_score_img*255
    # norm_sig_score_img = cv2.resize(norm_sig_score_img, (scale,scale))
    # plt.imshow(norm_sig_score_img) 
    # plt.show()
    x, y = np.where(sig_score_img == ma)
    x=x.item()
    y=y.item()
    src_img=image
    


    rect_img=np.zeros((sig_score_img.shape[0],norm_sig_score_img.shape[1]))
    rect_img[x-math.floor(region_size[0]/2):x+math.floor(region_size[0]/2),y-math.floor(region_size[1]/2):y+math.floor(region_size[1]/2)]=128
    
    
    a=(image.shape[0]/60)*60
    b=(image.shape[1]/60)*60
    
    

    rect_img=cv2.resize(rect_img, (int(b),int(a)))
    
    rect_size=[image.shape[0], image.shape[1]]
    src_img[:,:,0] = src_img[:,:,0] + np.uint8(rect_img[0:rect_size[0],0:rect_size[1]])
    # src_img = cv2.resize(src_img, (500,500))
  
    # cv2.imshow("Resized (Height) ", src_img)
    # cv2.waitKey(0)
    random_number = random.randint(1, 1235)
    status = cv2.imwrite(f'./out/{random_number}.jpg', image)
    

    
    # cv2.imshow("Resized (Height) ", norm_sig_score_img)
    # cv2.waitKey(0)
   
    
    # print(norm_sig_score_img[0][42])
    # # position = find_element_2d(norm_sig_score_img, ma)
    # print((ma))
    # # print(sig_score_img[0][42])
    
    # print(position)
    # print(norm_sig_score_img.shape)
    
    
    # color = (255, 0, 0)
    # thickness = 5

region_size = [45, 37]
patch_size = [5, 5]

radius, angle = cart2polar(region_size)

bin = getbin(radius, angle, region_size)




# image1 = cv2.imread('1.jpg')
# image2 = cv2.imread("2.jpg")

# imgRgb1=cv2.resize(image1, (0, 0), fx = 1/5, fy = 1/5)
# imgRgb2 = cv2.resize(image2, (0, 0), fx = 1/5, fy = 1/5)
# imgRgb1=cv2.resize(image1, (60, 60))
# imgRgb2=cv2.resize(image2, (60, 60))

# print(imgRgb1.shape,'img1', imgRgb2.shape,'img2')
# cv2.imshow("lalala", imgRgb1) 
# cv2.waitKey(0)



n_img=5
for i in range(n_img):
    name = f'{i+1}.jpg'
    # print(name)
    image = cv2.imread(f'{name}', 1)
 
   
    print(name)
    
    print(image.shape)
  
    imgRgb=cv2.resize(image, (60, 60))
    
    # cv2.imshow("lalala", imgRgb) 
    # cv2.waitKey(0)
    self_similarities=com_Self_Similarities(imgRgb,region_size,patch_size,bin)
    # print(self_similarities)
    np.save(f'mat{i+1}', self_similarities)
    
    
    
    


# self_similarities1 = com_Self_Similarities(imgRgb1, region_size, patch_size, bin)




# self_similarities2 = com_Self_Similarities(imgRgb2, region_size, patch_size, bin)
# print(self_similarities2)
  


width = 1
height = 1
center_sub = [math.floor(width/2), math.floor(height/2)]
p = 1
self_similarities1 = np.zeros((1, n_img))
for m in range(n_img):
    name = f'mat{m+1}.npy'
    data = np.load(f'{name}')
    # print(data)
    # src_img = cv2.imread(f'{name}', 1)
    img_size1 = data.shape
    name = f'{m+1}.jpg'
    self_similarities1 = data
    src_img = cv2.imread(f'{name}', 1)
    # imgRgb = cv2.resize(src_img, (0, 0), fx=1/3, fy=1/3)
    # imgRgb=cv2.resize(src_img, (60,60))
    sig_score_img = np.zeros((img_size1[0], img_size1[1]))
    temp_score=np.array((n_img))
    for row1 in range(center_sub[0], img_size1[0]-center_sub[0]):
        for col1 in range(center_sub[1], img_size1[1]-center_sub[1]):
            print(row1,col1)
            sub1 = self_similarities1[row1 - center_sub[0]:row1 +center_sub[0]+1, col1 - center_sub[1]:col1 + center_sub[1]+1, :]
            max_match=np.zeros((n_img-1))
            match_score={}
            num_img=0
            for n in range(n_img):
                if n!=m:
                    Path = (f'mat{n+1}.npy')
                    temp = np.load(Path)
                    self_similarities2 = temp
                    temp1 = np.tile(sub1, (self_similarities2.shape[0], self_similarities2.shape[1], 1))
                    temp2 = -1 * np.sum((self_similarities2 - temp1) ** 2, axis=2)
                    # print((np.max(np.max(temp2))))
                    # exit()
                    max_match[num_img]=np.max(np.max(temp2))
                 
                    temp2=temp2.reshape(1,-1)
                    match_score[num_img]=temp2
                    
                 
                    num_img=num_img+1
            
            # temp3=match_score;
           
            avgMatch=np.zeros((num_img))
            stdMatch=np.zeros((num_img))
         
            for i in range(n_img-1):
                

                
                # c=0
                
                # for j in match_score[i][0]:
                #     if(j==-0):
                #         print(j)
                #         # match_score[i][0][c]=0
                #         # c=c+1
                # print(np.std(match_score[i][0]))
                # exit()
                avgMatch[i]=(np.mean(match_score[i][0]))
                stdMatch[i]=np.std(match_score[i])
             
          
            sig_score_img[row1,col1] = sum((max_match-avgMatch)/stdMatch)
        
    # print(sig_score_img.shape)
    # for i in range(sig_score_img.shape[0]):
    #     for j in range(sig_score_img.shape[1]):
    #         print(sig_score_img[i][j])
    # exit()
            

    dr(src_img, sig_score_img/4, region_size,3)
  
    # print(x.item())
                   
    # print(ma)        
        
                    
                        

    
   
    
    
    
    
    
exit()    
# img_size1 = self_similarities1.shape

# print(imgRgb1.shape)
# print(self_similarities1.shape,"s")

# self_similarities={}
# sig_score_img = np.zeros((img_size1[0], img_size1[1]))
# for m in range(n_img):



# ma =np.max(np.max(sig_score_img))
# mi=np.min(np.min(sig_score_img))

# arr=sig_score_img
# max_value=0
# positions = []
# for i in range(arr.shape[0]):  # Loop through rows
#         for j in range(arr.shape[1]):  # Loop through columns
#             if arr[i, j] == max_value:
#                 positions.append((i, j))

# print(positions)
# norm_sig_score_img = (sig_score_img - mi)/(ma-mi)
# norm_sig_score_img = norm_sig_score_img*255



# print(norm_sig_score_img.shape)
# x = find_element_2d(sig_score_img, ma)
# x, y = np.where(sig_score_img == ma)
# print(x)







# for row1 in range(center_sub[0], img_size1[0]-center_sub[0]):
    
    # for col1 in range(center_sub[1], img_size1[1]-center_sub[1]):
    #     # print(row1,col1)
    #     max_match = {}
    #     match_score={}
    #     sub1 = self_similarities1[row1 - center_sub[0]:row1 +center_sub[0]+1, col1 - center_sub[1]:col1 + center_sub[1]+1, :]
    #     temp1 = np.tile(sub1, (self_similarities2.shape[0], self_similarities2.shape[1], 1))
        # temp2 = -1 * np.sum((self_similarities2 - temp1) ** 2, axis=2)
        # print(temp2.shape)
        # exit()
        # max_match[1] = np.max(np.max(temp2))
        # print(max_match)
        # match_score[1] = (temp2)
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