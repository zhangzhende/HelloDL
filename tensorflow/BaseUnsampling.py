from myProject.tensorflowDemo.slimAdvance.datasets import flowers
from myProject.tensorflowDemo.slimAdvance.datasets import imagenet_classes
from myProject.tensorflowDemo.slimAdvance import configure
from myProject.tensorflowDemo.slimAdvance.preprocessing import vgg_preprocessing
from myProject.tensorflowDemo.slimAdvance.nets import vgg as model
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.python.slim.data.dataset_data_provider as provider
import os
import numpy as np
from skimage import io
from numpy import ogrid,repeat,newaxis
"""
FCN全卷积神经网络部分--描述反卷积（上卷积）
"""

imageSize=224
checkPointPath=configure.Vgg16ModelPath

def preImage():
    """
    代码有问题，会报错，
    :return: 
    """
    #TODO
    with tf.Graph().as_default():
        imagePath="../jpg/001/1.jpg"
        image=tf.image.decode_jpeg(tf.read_file(imagePath),channels=3)
        image=vgg_preprocessing.preprocess_image(image=image,output_height=imageSize,output_width=imageSize,is_training=False)
        image=tf.expand_dims(image,0)

        with slim.arg_scope(model.vgg_arg_scope()):
            logits,_=model.vgg_16(inputs=image,is_training=False)
            prob=slim.softmax(logits=logits)

        modelPath=os.path.join(configure.Vgg16ModelPath,configure.Vgg16ModelName)
        init_fn=slim.assign_from_checkpoint_fn(model_path=modelPath,var_list=slim.get_model_variables("vgg_16"))

        with tf.Session() as sess:
            init_fn(sess)
            res=sess.run([prob])
            prob=res[0]
            sortedIds=[i for i in sorted(enumerate(-prob),key=lambda x:x[1])]
            names=imagenet_classes.class_names
            for i in range(5):
                index=sortedIds[i]
                print(index)
                print(sortedIds)
                print("probability %0.2f =[%s]"%(prob[index],names[index]))
# preImage()

"""
21-2
创建一个【3,3】的黑图案
"""
def drawPic():
    size=3
    x,y=ogrid[:size,:size]
    img=repeat((x+y)[...,newaxis],3,2)/12.
    io.imshow(img,interpolation="none")
    io.show()
# drawPic()
"""
21-3
反卷积处理
"""
def upsampling():
    """
    上取样
    :return: 
    """
    size = 3
    x, y = ogrid[:size, :size]
    img = repeat((x + y)[..., newaxis], 3, 2) / 12.
    img=tf.cast(img,tf.float32)
    img=tf.expand_dims(img,0)#扩大一个维度
    kernel=tf.random_normal([5,5,3,3],dtype=tf.float32)
    #反卷积
    res=tf.nn.conv2d_transpose(img,kernel,[1,9,9,3],[1,3,3,1],padding="SAME")
    with tf.Session() as sess:
        img=sess.run(tf.squeeze(res))#使用图进行

    io.imshow(img/np.argmax(img),interpolation="none")#显示压缩后图像
    io.show()

upsampling()

"""
下面三个方法表示的是双线性插值
"""
#确定卷积核大小
def getKernelSize(factor):
    return 2*factor-factor%2
#计算
def upsampleFilt(size):
    factor=(size+1)//2#整除
    if size % 2==1:
        center=factor-1
    else:
        center=factor-0.5
    og=np.ogrid[:size,:size]
    return (1-abs(og[0]-center)/factor)*(1-abs(og[1]-center)/factor)
#进行upsampling 卷积
def bilinear_upsample_weight(factor,number_of_class):
    filter_size=getKernelSize(factor=factor)
    weights=np.zeros((
        filter_size,
        filter_size,
        number_of_class,
        number_of_class
    ),dtype=np.float32)
    upsample_kernel=upsampleFilt(filter_size)
    for i in range(number_of_class):
        weights[:,:,i,i]=upsample_kernel
        # print(weights[:,:,i,i])
    return weights

def doubleLineIn():
    """
    双线性差值反卷积恢复图片
    图片有问题，一篇漆黑
    :return: 
    """
    size=3
    x,y=ogrid[:size,:size]
    img=repeat((x+y)[...,newaxis],3,2)/12.
    kernel=bilinear_upsample_weight(3,3)
    res=tf.nn.conv2d_transpose(img,kernel,[1,9,9,3],[1,3,3,1],padding="SAME")
    with tf.Session() as sess:
        img=sess.run(tf.squeeze(res))
        print(img)
    io.imshow(img)
    io.show()

doubleLineIn()




