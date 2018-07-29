import numpy as np
import tensorflow as tf
from HelloDL.Nin import NINModel as model
from HelloDL.Nin import CreateAndReadTFRecords as reader2
import time
import os
from scipy.misc import imread,imresize


"""

还有问题，计算报错：
logits and labels must be broadcastable: logits_size=[80000,2] labels_size=[10,2]
"""
#TODO
def trainModel():
    imageList,labelList=reader2.getFile("../CatVsDog/catanddog")
    imageBatch,labelBatch=reader2.getBatch(imageList=imageList,labelList=labelList,imgWidth=224,imgHeight=224,batchSize=25,capacity=256)
    xInput=tf.placeholder(dtype=tf.float32,shape=[None,224,224,3])
    yInput=tf.placeholder(dtype=tf.float32,shape=[None,2])

    models=model.NINModel(xInput)
    ninOut=models.result
    probs = tf.nn.softmax(ninOut)
    loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=probs,labels=yInput))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(loss)
    correct_pred = tf.equal(tf.argmax(ninOut, 1), tf.argmax(yInput, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    sess=tf.Session()
    sess.run(tf.global_variables_initializer())
    saver =models.saver()

    coord=tf.train.Coordinator()
    threads=tf.train.start_queue_runners(coord=coord,sess=sess)

    startTime=time.time()
    for i in range(10):
        image ,label=sess.run([imageBatch,labelBatch])
        labels=reader2.onehot(label)

        sess.run(optimizer,feed_dict={xInput:image,yInput:labels})
        lossRecord=sess.run(loss,feed_dict={xInput:image,yInput:labels})
        print("now the loss is %f "%lossRecord)
        endTime=time.time()
        print("time:",(endTime-startTime))
        startTime=endTime
        print("----------epoch %d is finished------------------"%i)

    saver.save(sess=sess,save_path="./model/")
    print("Optimization Finished!!")
    coord.request_stop()
    coord.join(threads)

def trainModel2():
    xTrain,yTrain=reader2.getFile("../CatVsDog/catanddog")
    imageBatch,labelBatch=reader2.getBatch(imageList=xTrain,labelList=yTrain,imgWidth=224,imgHeight=224,batchSize=25,capacity=256)
    xImgs=tf.placeholder(tf.float32,[None,224,224,3])
    yImgs=tf.placeholder(tf.float32,[None,2])
    vgg=model.NINModel(xImgs)
    ninOut = vgg.NINResult
    # fc3CatAndDog=tf.nn.softmax(ninOut)
    loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=ninOut,labels=yImgs))
    optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(loss)

    sess=tf.Session()
    sess.run(tf.global_variables_initializer())
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

    saver.save(sess=sess,save_path="./model/")
    print("Optimization Finished!!")
    coord.request_stop()
    coord.join(threads)
# trainModel()
trainModel2()