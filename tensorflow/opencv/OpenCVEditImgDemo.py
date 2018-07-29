"""
图片的裁剪编辑等
"""
import numpy as np
import cv2
import matplotlib.pyplot as plt
import random
"""
图片的缩放cv2.resize(src, dsize, dst=None, fx=None, fy=None, interpolation=None)
src=目标图片
dsize=缩放大小
"""
image = cv2.imread("./img/58.jpg", cv2.IMREAD_COLOR)
def useResize():
    result=cv2.resize(image,(500,600))
    cv2.imshow("cool.png", result)
    cv2.waitKey(0)
"""
图片裁剪，安坐标裁剪图片
"""
def imgCaijian():
    patch_tree=image[10:500,10:600]
    cv2.imshow("cool.png", patch_tree)
    cv2.waitKey(0)
"""
HSV:H指的是色调【0-180】，S指的是饱和度【0-255】，V指的是明暗度【0-255】
turn_green_hsv[:,:,0]，0表示色调，1表示饱和度，2表示明暗度
"""
"""
对于：(turn_green_hsv[:,:,0]-30)%180，结果会是正数
Python 语言除法采用的是 floor 除法，所以对 Python 程序员来讲：
-17 % 10 的计算结果如下：r = (-17) - (-17 / 10) x 10 = (-17) - (-2 x 10) = 3
17 % -10 的计算结果如下：r = 17 - (17 / -10) x (-10) = (17) - (-2 x -10) = －3
-17 % -10 的计算结果如下：r = (-17) - (-17 / -10) x (-10) = (-17) - (1 x -10) = -7
据说，Python 3.x 中「/」运算符的意义发生了变化，「/」产生的结果将不会再进行取整，相应的「//」运算符的结果才会进行取整。
"""
def imgSediao():
    img_hsv=cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    turn_green_hsv=img_hsv.copy()
    #色调调整
    # turn_green_hsv[:,:,0]=(turn_green_hsv[:,:,0]-30)%180
    turn_green_hsv[:, :, 0] = turn_green_hsv[:, :, 0]*0.6
    #饱和度调整
    # turn_green_hsv[:, :, 1] = turn_green_hsv[:, :, 1] *0.6
    #明暗度调整
    # turn_green_hsv[:, :, 2] = turn_green_hsv[:, :, 2]  *0.6
    turn_green_img=cv2.cvtColor(turn_green_hsv,cv2.COLOR_HSV2BGR)
    cv2.imshow("cool.png", turn_green_img)
    cv2.waitKey(0)

"""
Gamma计算
LUT的作用，0-100对应0,100-200对应2 ，200+对应3，这样修改各像素的灰度值
把图像中的数据从之前的比较高的灰度级降下来，例如灰度级是256的char类型的灰度级，
我们通过一个参数（例如上述程序中就是100），将原来的256个灰度级降到了3个灰度级，
原来图像中灰度值在0-100的数据现在灰度值变成了0，原来灰度值为101-200的图像数据
现在灰度值变为了1，而201-256的灰度值就变为了2。所以通过参数100，图像的灰度级就
到了2，只有0，1，2三个灰度值，那么原来的图像矩阵中的每一位数据我们是char型的，
需要8位来表示一个数据，而灰度级降下来之后，我们只需要2位就足以表示所有灰度值。
"""
def useGamma():
    img=plt.imread("./img/58.jpg")
    """
    power(x1, x2)
    对x1中的每个元素求x2次方。
    """
    gamma_change=[np.power(x/255,0.4)* 255 for x in range(256)]
    gamma_img=np.round(np.array(gamma_change)).astype(np.uint8)
    print(gamma_img)
    img_corrected=cv2.LUT(img,gamma_img)
    #subplot（2,2,1）,表示2*2的面板取第1个
    plt.subplot(221)
    plt.imshow(img)
    plt.subplot(222)
    plt.imshow(img_corrected)
    plt.show()
"""
图像的旋转，平移，和反转
"""
def imgpingyi():
    #仿射变换矩阵,逆时针旋转90度，并缩放0.8,横坐标左移100，纵坐标下移12
    # M_copy_img=np.array([[0,0.8,-100],[0.8,0,-12]],dtype=np.float32)
    # 仿射变换矩阵,缩放0.8,横坐标左移100，纵坐标下移12
    M_copy_img = np.array([[0.8, 0, -100], [0, 0.8, -12]], dtype=np.float32)
    img_change=cv2.warpAffine(image,M_copy_img,(300,500))
    cv2.imshow("test",img_change)
    cv2.waitKey(0)

"""
图片的随机裁剪，选好矩形起始点的随机变换范围，然后在其中产生随机数，然后对应裁剪
"""
def suijicaijian():
    width,height,depth=image.shape
    img_width_box=width*0.2
    img_height_box=height*0.2
    for _ in range(2):
        start_pointX=(int)(random.uniform(0,img_width_box))
        start_pointY=(int)(random.uniform(0,img_height_box))
        #start_pointX,start_pointY必须得是整数，所以上面进行了强制类型转换
        copyImg=image[start_pointX:200,start_pointY:200]
        cv2.imshow("test",copyImg)
        cv2.waitKey(0)
"""
随机旋转变换
getRotationMatrix2D(center, angle, scale)函数调用返回一个仿射矩阵
@center 以哪一点为原点进行选择
@angle 旋转角度大小
@scale 缩放的倍数
"""
def suijixuanzhaun():
    rows,cols,depth=image.shape
    #构造仿射矩阵
    image_change=cv2.getRotationMatrix2D((cols/2,rows/2),90,2)
    print(image_change)
    res=cv2.warpAffine(image,image_change,(rows,cols))
    cv2.imshow("test", res)
    cv2.waitKey(0)

"""
图像色彩的随机变换
"""
def suijisecai():
    img_hsv=cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    turn_green_hsv=img_hsv.copy()
    # np.random.randint(180)，生成0-180之间的随机数
    turn_green_hsv[:,:,0]=(turn_green_hsv[:,:,0]+np.random.randint(0,180))%180
    turn_green_hsv[:, :, 1] = (turn_green_hsv[:, :, 1] + np.random.randint(180)) % 180
    turn_green_hsv[:, :, 2] = (turn_green_hsv[:, :, 2] + np.random.randint(180)) % 180
    turn_green_img=cv2.cvtColor(turn_green_hsv,cv2.COLOR_HSV2BGR)
    cv2.imshow("test", turn_green_img)
    cv2.waitKey(0)
start = (0, 0)
rect_end = (0, 0)
"""
跟踪鼠标事件，绘制矩形
"""
def on_mouse(event,x,y,flags,param):
    #设置使用全局参数，否则每次调用都会重置参数
    global rect_start
    global rect_end
    if event ==cv2.EVENT_LBUTTONDOWN:
        rect_start=(x,y)
        print(rect_start)
    elif event==cv2.EVENT_LBUTTONUP:
        rect_end=(x,y)
        print(rect_end)
        cv2.rectangle(image,rect_start,rect_end,(0,255,0),2)
def getMouseEvent():
    cv2.namedWindow('test')
    cv2.setMouseCallback("test",on_mouse)
    while(1):
        cv2.imshow("test",image)
        if cv2.waitKey(1)&0xFF==ord('q'):
            break
    cv2.destroyAllWindows()


# useResize()
# imgCaijian()
# imgSediao()
# useGamma()
# imgpingyi()
# suijicaijian()
# suijixuanzhaun()
# suijisecai()
getMouseEvent()