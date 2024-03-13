import numpy as np
import math
import cv2
import matplotlib.pyplot as plt

def show(img):
    cv2.imshow('img',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
size = 3
pi = 3.1415926
def make_guasstamplate(n = size,sigma = 1):
    #阶数   n,默认为size=3  ，方差   d = sigma^2默认等于1
    #创建初始化二维矩阵guass
    
    print("初始化高斯模板")

    guass = np.zeros((n,n))

    # array = np.zeros((n,n))
    if n % 2 == 1 :     #判断是否为奇数
        mid = (n-1)/2
        for i in range(n):
            for j in range(n):
                #高斯滤波
                p = np.exp(-((math.pow(i-mid,2) + math.pow(j-mid,2)) / (2*math.pow(sigma,2)))) / (2 * np.pi * math.pow(sigma,2))
                guass[i][j] = p
   
    #对高斯模板归一化
                
    # print("高斯模板未归一：",guass)
    # print("高斯模板求和：",np.sum(guass))
    guass = guass / np.sum(guass)
    print("高斯模板：",guass)
    return guass

def gaussfilter(img,n,sigma):
    h=img.shape[0]
    w=img.shape[1]
    print(h,type(h))
    img1=np.zeros((h,w),np.uint8)
    kernel=make_guasstamplate(n,sigma)   # 计算高斯卷积核
    mid = (n-1)/2
    mid = int(mid)
    for i in range (mid,h-mid):     #去掉最外围一圈
        for j in range (mid,w-mid):
            sum = 0
            for k in range(-mid,mid+1):       
                for l in range(-mid,mid+1):
                    sum+=img[i+k,j+l]*kernel[k+mid,l+mid]   # 高斯滤波
            img1[i,j]=sum
    return img1

if __name__ == '__main__' :
#灰度高斯滤波
    # img = cv2.imread("D:\pytest\hw1\image\girl1.png",cv2.IMREAD_GRAYSCALE)  #2d
    # print(img)
    # img_1= gaussfilter(img,5,1.5)
    # print(img_1)
    # cv2.imshow('img',img)
    # cv2.imshow('img_1',img_1)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    
#RBG三通道高斯滤波
    img = cv2.imread("D:\pytest\hw1\image\girl2.png")  #RBG
    print(img)
    b, g, r = cv2.split(img)
    b_blur = gaussfilter(b,5,3)
    g_blur = gaussfilter(g,5,3)
    r_blur = gaussfilter(r,5,3)
    img_blur_1 = cv2.merge((b_blur, g_blur, r_blur))
    cv2.imshow('img',img)
    cv2.imshow('B_img',img_blur_1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()