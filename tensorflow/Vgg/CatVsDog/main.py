import numpy as np
import tensorflow as tf
from myProject.tensorflowDemo.VGG16.save_and_restore.CatVSDogWithVggnet import Vgg16Model as model
from myProject.tensorflowDemo.VGG16.save_and_restore.CatVSDogWithVggnet import CreateAndReadTFRecords as reader2
import time
import os
from scipy.misc import imread,imresize

"""
16-15训练模型的方法
"""
def trainModel():
    xTrain,yTrain=reader2.getFile("../../../CatVsDog/catanddog")
    imageBatch,labelBatch=reader2.getBatch(imageList=xTrain,labelList=yTrain,imgWidth=224,imgHeight=224,batchSize=25,capacity=256)
    xImgs=tf.placeholder(tf.float32,[None,224,224,3])
    yImgs=tf.placeholder(tf.float32,[None,2])
    vgg=model.vgg16(xImgs)
    fc3CatAndDog=vgg.probs
    loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=fc3CatAndDog,labels=yImgs))
    optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(loss)

    sess=tf.Session()
    sess.run(tf.global_variables_initializer())
    vgg.loadWeights("../data/vgg16_weights.npz",sess=sess)
    saver =vgg.saver()

    coord=tf.train.Coordinator()
    threads=tf.train.start_queue_runners(coord=coord,sess=sess)

    startTime=time.time()
    for i in range(10):
        image ,label=sess.run([imageBatch,labelBatch])
        labels=reader2.onehot(label)

        sess.run(optimizer,feed_dict={xImgs:image,yImgs:labels})
        lossRecord=sess.run(loss,feed_dict={xImgs:image,yImgs:labels})
        print("now the loss is %f "%lossRecord)
        endTime=time.time()
        print("time:",(endTime-startTime))
        startTime=endTime
        print("----------epoch %d is finished------------------"%i)

    saver.save(sess=sess,save_path="./VggFinetuningModel/")
    print("Optimization Finished!!")
    coord.request_stop()
    coord.join(threads)

"""
复用上面保存的模型，预测图片
"""
def modelReuse():
    imgs=tf.placeholder(tf.float32,[None,224,224,3])
    sess=tf.Session()
    vgg=model.vgg16(imgs)
    fc3CatAndDog=vgg.probs
    saver =vgg.saver()
    saver.restore(sess=sess,save_path="./VggFinetuningModel/")
    for root,subFolders,files in os.walk("../../../CatVsDog/datatest"):
        i=0
        cat=0
        dog=0
        for name in files:
            i+=1
            filepath=os.path.join(root,name)
            try:
                img1=imread(filepath,mode="RGB")
                img1=imresize(img1,(224,224))
            except:
                print("remove",filepath)
            prob=sess.run(fc3CatAndDog,feed_dict={vgg.imgs:[img1]})[0]
            maxIndex=np.argmax(prob)
            if maxIndex==0:
                print(name,":cat!!")
                cat+=1
            else:
                print(name, ":dog!!")
                dog += 1
            if i%50==0:
                acc=(dog*1.)/(cat+dog)
                print("acc:",acc)
                print("------image number is %d-------"%i)


# trainModel()
modelReuse()

















































