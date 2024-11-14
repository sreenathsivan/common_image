import math
import numpy as np


def getbin(radius, angle, region_size):




    max_radius = np.max(radius)
   
    # print(radius[0])
    bin = []
    # bin=np.array((20,4))
    # temp=[]
    for m in range(20):
        row = []
        for n in range(4):
            row.append([])
        bin.append(row)
        # temp.append(row)

    for m in range(20):  # theta
        theta_low = m*18
        theta_up = (m+1)*18
        
        for n in range(4):  # radius
            rho_low = max_radius*n/4
            rho_up = max_radius*(n+1)/4
  
            temp = []
            num = 0

            for row in range(region_size[0]):
                for col in range(region_size[1]):

                    if (radius[row][col] >= rho_low) and (radius[row][col] <= rho_up) and (angle[row][col] >= theta_low) and (angle[row][col] <= theta_up):
                        temp.append((row, col))

            bin[m][n] = temp

            # print(bin)

    # print((bin[0][]))
    # exit()
    # f=open("test.txt","a")
    # f.write(str(bin))
    # f.close()
    # exit()

    return bin
