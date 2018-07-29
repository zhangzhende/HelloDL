import tensorflow as tf
import myProject.tensorflowDemo.VGG16.save_and_restore.VGGNET1.Vgg16ClassNew as model
import myProject.tensorflowDemo.VGG16.save_and_restore.VGGNET2.global_variable as global_variable
from myProject.tensorflowDemo.VGG16.save_and_restore.common.imagenet_classes import class_names
from scipy.misc import imread,imresize
import numpy as np

"""
16-11
保存复用的VGGNet模型为TensorFlow格式
"""
def saveModel():
    imgs=tf.placeholder(tf.float32,shape=[None,224,224,3])
    vgg=model.vgg16(imgs)
    prob=vgg.probs
    saver=vgg.saver()
    sess=tf.Session()
    vgg.loadWeights("../data/vgg16_weights.npz",sess=sess)
    saver.save(sess=sess,save_path=global_variable.save_path)

"""
16-12
复用上面保存好的TensorFlow格式的文件
"""
def useModel():
    imgs = tf.placeholder(tf.float32, [None, 224, 224, 3])
    vgg = model.vgg16(imgs=imgs)
    saver =vgg.saver()
    sess = tf.Session()
    saver.restore(sess=sess,save_path=global_variable.save_path)
    img1 = imread("../data/cat14.jpg", mode="RGB")
    img1 = imresize(img1, (224, 224))
    prob = sess.run(vgg.probs,feed_dict={vgg.imgs:[img1]})[0]
    preds = (np.argsort(prob)[::-1])[0:5]
    for p in preds:
        print(class_names[p], prob[p])


# saveModel()
useModel()