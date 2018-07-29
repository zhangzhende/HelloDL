import tensorflow as tf
import HelloDL.resnet.Tool as tool
import time
n_class=2

"""
Resnet，残差网络的简单认识

残差网络与传统的网络相比，最大的变化就是加入了一个恒等映射层，y=x，
"""


"""
训练方法"""
def train(inputData):
    outputData=tool.batchNorm(inputData)
    outputData=tool.conv(layerName="first_layer",inputData=outputData,outChannel=32,kernelSize=[7,7])
    outputData=tool.maxPool(input=outputData,kernelHeight=3,kernelWidth=3,strideHeight=2,strideWight=2,name="maxPool1")

    outputData=tool.resUnit(inputData=outputData,outChannel=[16,16,32],i=0)

    outputData=tool.avgPool(outputData,kernelHeight=7,kernelWidth=7,strideHeight=1,strideWight=1,name="avgPool")

    res=tool.fc(name="fc",inputData=outputData,outChannel=n_class)
    res=tf.nn.softmax(res)
    print("model finished!!")
    return res

def trainModel():
    imageList,labelList=tool.getFile("../CatVsDog/catanddog")
    imageBatch,labelBatch=tool.getBatch(imageList=imageList,labelList=labelList,imgWidth=224,imgHeight=224,batchSize=25,capacity=256)
    xInput=tf.placeholder(dtype=tf.float32,shape=[None,224,224,3])
    yInput=tf.placeholder(dtype=tf.float32,shape=[None,2])
    probs = train(xInput)
    loss=tool.getLoss(probs=probs,yInput=yInput)
    optimizer = tool.getOptimizer(loss=loss,learningRate=0.01)
    correct_pred = tf.equal(tf.argmax(probs, 1), tf.argmax(yInput, 1))
    accuracy = tool.getAccuracy(correct_pred)

    sess=tf.Session()
    sess.run(tf.global_variables_initializer())
    saver =tool.saver()

    coord=tf.train.Coordinator()
    threads=tf.train.start_queue_runners(coord=coord,sess=sess)

    startTime=time.time()
    for i in range(10):
        image ,label=sess.run([imageBatch,labelBatch])
        labels=tool.onehot(label)

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