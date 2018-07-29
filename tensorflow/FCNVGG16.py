import os
from matplotlib import pyplot as plt
import numpy as np
from skimage import io
from numpy import ogrid, repeat, newaxis
import tensorflow as tf
from myProject.tensorflowDemo.slimAdvance import configure
import myProject.tensorflowDemo.slimAdvance.datasets.imagenet_classes  as imagenet_classes
from tensorflow.contrib.slim.nets import vgg
import tensorflow.contrib.slim as slim
from myProject.tensorflowDemo.slimAdvance.preprocessing .vgg_preprocessing import (_mean_image_subtraction,_R_MEAN,_G_MEAN,_B_MEAN)

"""
全卷积神经网络--使用VGG16模型进行图像识别
21-6
"""
# 确定卷积核大小
def getKernelSize(factor):
    return 2 * factor - factor % 2


def upsampleFilt(size):
    factor = (size + 1) // 2  # 整除
    if size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:size, :size]
    return (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)


# 进行upsampling 卷积
def bilinear_upsample_weight(factor, number_of_class):
    filter_size = getKernelSize(factor=factor)
    weights = np.zeros((
        filter_size,
        filter_size,
        number_of_class,
        number_of_class
    ), dtype=np.float32)
    upsample_kernel = upsampleFilt(filter_size)
    for i in range(number_of_class):
        weights[:, :, i, i] = upsample_kernel
        # print(weights[:,:,i,i])
    return weights


def upsample_tf(factor,input_img):
    number_of_classes=input_img.shape[2]
    new_height=input_img.shape[0]*factor
    new_width=input_img.shape[1]*factor
    expanded_img=np.expand_dims(input_img,axis=0)

    with tf.Graph().as_default():
        with tf.Session() as sess:
            upsample_filter_np=bilinear_upsample_weight(factor,number_of_classes)
            res=tf.nn.conv2d_transpose(expanded_img,upsample_filter_np,
                                       output_shape=[1,new_height,new_width,number_of_classes],
                                       strides=[1,factor,factor,1])
            res=sess.run(res)
    return np.squeeze(res)

def discrete_matshow(data,labels_names=[],title=""):
    fig_size=[7,6]
    plt.rcParams["figure.figsize"]=fig_size
    cmap=plt.get_cmap("Paired",np.max(data)-np.min(data)+1)
    mat=plt.matshow(data,cmap=cmap,vmin=np.min(data)-0.5,vmax=np.max(data)+0.5)
    cax=plt.colorbar(mat,ticks=np.arange(np.min(data),np.max(data)+1))
    if labels_names:
        cax.ax.set_yticklabels(labels_names)
    if title:
        plt.suptitle(title,fontsize=15,fontweight="bold")
        plt.show()

def imageBreak():
    imagepath="../../data/beauty/001/2.jpg"
    with tf.Graph().as_default():
        image=tf.image.decode_jpeg(tf.read_file(imagepath),channels=3)
        image=tf.image.resize_images(image,[224,224])
        #减去均值之前，将像素转为32位浮点数
        image_float=tf.to_float(image,name="ToFloat")
        #每个像素减去像素的均值
        processed_image=_mean_image_subtraction(image_float,[_R_MEAN,_G_MEAN,_B_MEAN])
        input_image=tf.expand_dims(processed_image,0)
        with slim.arg_scope(vgg.vgg_arg_scope()):
            logits,endpoints=vgg.vgg_16(inputs=input_image,
                                        num_classes=1000,
                                        is_training=False,
                                        spatial_squeeze=False)
            pred=tf.argmax(logits,dimension=3)#对输出层进行逐个比较，取得不同层同一位置中最大的概率所对应的值
            init_fn=slim.assign_from_checkpoint_fn(
                os.path.join(configure.Vgg16ModelPath,configure.Vgg16ModelName),
                slim.get_model_variables("vgg_16")
            )
            with tf.Session() as sess:
                init_fn(sess)
                fcn8s,fcn16s,fcn32s=sess.run([endpoints["vgg_16/pool3"],endpoints["vgg_16/pool4"],endpoints["vgg_16/pool5"]])
                upsampled_logits=upsample_tf(factor=16,input_img=fcn8s.squeeze())
                upsampled_predictions32=upsampled_logits.squeeze().argmax(2)

                unique_classes,relabeled_image=np.unique(upsampled_predictions32,return_inverse=True)
                relabeled_image=relabeled_image.reshape(upsampled_predictions32.shape)

                labels_names=[]
                names=imagenet_classes.class_names
                for index,current_class_number in enumerate(unique_classes):
                    labels_names.append(str(index)+" "+names[current_class_number+1])
                discrete_matshow(data=relabeled_image,labels_names=labels_names,title="Segmentation")

imageBreak()