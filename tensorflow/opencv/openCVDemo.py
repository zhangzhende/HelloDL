"""

openCV的简单使用
"""
import numpy as np
import cv2
import os
from scipy import ndimage
"""定义一个矩阵然后作为图片展示"""
def drawMatrixImg():
    img = np.mat(np.zeros((300, 300)))
    cv2.imwrite("./data/3.png", img)
    cv2.imshow("test", img)
    cv2.waitKey(0)


"""图片的读取与写入"""
def imgreadwrite():
    # cv2.IMREAD_GRAYSCALE=0表示返回一个灰度图像
    # image=cv2.imread("./img/2.jpg",cv2.IMREAD_GRAYSCALE)
    # cv2.IMREAD_ANYCOLOR=4表示返回一个彩色图像
    # image = cv2.imread("./img/58.jpg", cv2.IMREAD_ANYCOLOR)
    # cv2.IMREAD_ANYCOLOR=2表示返回一个深度图像
    # image = cv2.imread("./img/58.jpg", cv2.IMREAD_ANYDEPTH)
    # cv2.IMREAD_ANYCOLOR=1表示返回一个彩色图像
    image = cv2.imread("./img/58.jpg", cv2.IMREAD_COLOR)
    cv2.imwrite("./data/2.png", image)


"""矩阵转图像"""
def imgarr():
    image = np.mat(np.zeros((300, 300)))
    imageByteArray = bytearray(image)
    print(imageByteArray)
    imageBGR = np.array(image).reshape(300, 300)
    cv2.imshow("cool.png", imageBGR)
    cv2.waitKey(0)


"""随机生成数组然后重构为图像"""
def ramdmimg():
    randomByteArray = bytearray(os.urandom(120000))
    flatnumpyArray = np.array(randomByteArray).reshape(300, 400)
    cv2.imshow("cool", flatnumpyArray)
    cv2.waitKey(0)


"""
通过修改数组的值来编辑修改图片
"""
def editimg():
    img = np.zeros((300, 300))
    img[0:10, 0:10] = 255#左上角10*10位置为白色
    cv2.imshow("img", img)
    cv2.waitKey(0)
"""
编辑画两条白线
"""
def draw2line():
    img =np.zeros((300,300))
    img[:,10]=255
    img[10,:]=255
    cv2.imshow("img",img)
    # 给定一个时间等待用户按键触发，如果用户没有按下键，则继续后面的。如果时间为0 则无限等待用户
    #这表示图像展示一秒钟后结束
    cv2.waitKey(1000)


"""
卷积核是一种常见的图像处理工具
例如：kerne133=np.array([
[-1,-1,-1],
[-1,8,-1],
[-1,-1,-1]])这是一个3*3的卷积核，其作用就是计算中央像素与周围像素的亮度差，如果亮度差距过大，本身图像中央
亮度较小，那么经过卷积以后，中央像素的亮度会增加，即一个像素如果比周围像素更突出，则他会更加突出，提升自己亮度。
相反：
kerne133=np.array([
[1,1,1],
[1,-8,1],
[1,1,1]])，这个核就是减小中心像素的亮度，即如果一个像素比周围像素昏暗，则经过卷积他会更加昏暗
"""

def juanjiimg():
    kerne133 = np.array([
        [-1, -1, -1],
        [-1, 8, -1],
        [-1, -1, -1]])
    kerne133_D = np.array([
        [1, 1, 1],
        [1, -8, 1]])
    img =cv2.imread("./img/58.jpg", cv2.IMREAD_GRAYSCALE)
    #convolve卷积运算
    lightImg2=ndimage.convolve(img,kerne133)
    lightImg = ndimage.convolve(img, kerne133_D)
    # cv2.imshow("img",lightImg)
    cv2.imshow("img", lightImg2)
    cv2.waitKey(0)

"""
高斯模糊
def GaussianBlur(src, ksize, sigmaX, dst=None, sigmaY=None, borderType=None)
. InputArray src: 输入图像，可以是Mat类型，图像深度为CV_8U、CV_16U、CV_16S、CV_32F、CV_64F。 
. Size ksize: 高斯内核大小，这个尺寸与前面两个滤波kernel尺寸不同，ksize.width和ksize.height可以不相同但是这两个值必须为正奇数，如果这两个值为0，他们的值将由sigma计算。 
. double sigmaX: 高斯核函数在X方向上的标准偏差 
"""
def gaussimg():
    img = cv2.imread("./img/58.jpg", cv2.IMREAD_GRAYSCALE)
    blurred=cv2.GaussianBlur(img,(11,11),0)
    gaussimg=img-blurred
    cv2.imshow("img", gaussimg)
    cv2.waitKey(0)

"""
对于矩阵，卷积计算的实现
@dateMat 原矩阵
@kernel 卷积核
@return newMat ,卷积计算结果
"""
def consolve(dateMat,kernel):
    m,n=dateMat.shape
    km,kn=kernel.shape
    newMat=np.ones(((m-km+1),(n-kn+1)))
    tempMat=np.ones(((km),(kn)))
    for row in range(m-km+1):
        for col in range(n-kn+1):
            for m_k in range(km):
                for n_k in range(kn):
                    tempMat[m_k,n_k]=dateMat[(row+m_k),(col+n_k)]*kernel[m_k,n_k]
            newMat[row,col]=np.sum(tempMat)
    return newMat
def useConsolve():
    dateMat = cv2.imread("./img/58.jpg", cv2.IMREAD_GRAYSCALE)
    tempMat = np.array([
        [-1, -1, -1],
        [-1, 8, -1],
        [-1, -1, -1]])
    result=consolve(dateMat,tempMat)
    # print(result)
    cv2.imshow("img", result)
    cv2.waitKey(0)

def filter2Duse():
    dateMat = cv2.imread("./img/58.jpg", cv2.IMREAD_COLOR)
    tempMat = np.array([
        [-1, -1, -1],
        [-1, 8, -1],
        [-1, -1, -1]])
    img=cv2.filter2D(dateMat,-1,tempMat)
    cv2.imshow("img", img)
    cv2.waitKey(0)

# drawMatrixImg()
# imgreadwrite()
# imgarr()
# ramdmimg()
# editimg()
# draw2line()
# juanjiimg()
# gaussimg()
# useConsolve()
filter2Duse()