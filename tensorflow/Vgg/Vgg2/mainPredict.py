import tensorflow as tf
import numpy as np
import myProject.tensorflowDemo.VGG16.save_and_restore.VGGNET1.Vgg16ClassNew as model
from scipy.misc import imread,imresize
from myProject.tensorflowDemo.VGG16.save_and_restore.common.imagenet_classes import class_names

if __name__=="__main__":
    imgs=tf.placeholder(tf.float32,[None,224,224,3])
    vgg=model.vgg16(imgs=imgs)
    prob=vgg.probs
    sess=tf.Session()
    vgg.loadWeights("../data/vgg16_weights.npz",sess=sess)
    img1=imread("../data/cat57.jpg",mode="RGB")
    img1=imresize(img1,(224,224))

    prob=sess.run(vgg.probs,feed_dict={vgg.imgs:[img1]})[0]
    preds=(np.argsort(prob)[::-1])[0:5]
    for p in preds:
        print(class_names[p],prob[p])