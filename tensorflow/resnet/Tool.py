import tensorflow as tf
import numpy as np
import os

"""
卷积
"""


def conv(layerName, inputData, outChannel, kernelSize, stride=[1, 1, 1, 1]):
    inChannels = inputData.get_shape()[-1]
    with tf.variable_scope(layerName):
        weights = tf.get_variable(name="weights", shape=[kernelSize[0], kernelSize[1], inChannels, outChannel])
        biases = tf.get_variable(name="biases", shape=[outChannel], initializer=tf.constant_initializer(0.0))
        inputData = tf.nn.conv2d(input=inputData, filter=weights, strides=stride, padding="SAME", name="conv")
        inputData = tf.nn.bias_add(inputData, biases, name="bias_add")
        outData = tf.nn.relu(inputData, name="relu")
        return outData


"""
归一化
"""


def batchNorm(inputs, isTraining=True, isConvOut=True, decay=0.999):
    scale = tf.Variable(tf.ones([inputs.get_shape()[-1]]))
    beta = tf.Variable(tf.zeros([inputs.get_shape()[-1]]))
    popMean = tf.Variable(tf.zeros([inputs.get_shape()[-1]]), trainable=False)
    popVar = tf.Variable(tf.ones(inputs.get_shape()[-1]), trainable=False)
    if isTraining:
        if isConvOut:
            batchMean, batchVar = tf.nn.moments(inputs, [0, 1, 2])
        else:
            batchMean, batchVar = tf.nn.moments(inputs, [0])

        trainMean = tf.assign(popMean, popMean * decay + batchMean * (1 - decay))
        trainVar = tf.assign(popVar, popVar * decay + batchVar * (1 - decay))
        with tf.control_dependencies([trainMean, trainVar]):
            return tf.nn.batch_normalization(inputs, mean=batchMean, variance=batchVar, offset=beta, scale=scale,
                                             variance_epsilon=0.001)

    else:
        return tf.nn.batch_normalization(inputs, mean=popMean, variance=popVar, offset=beta, scale=scale,
                                         variance_epsilon=0.001)


"""
最大值池化

"""


def maxPool(input, kernelHeight, kernelWidth, strideHeight, strideWight, name, padding="SAME"):
    return tf.nn.max_pool(input=input, ksize=[1, kernelHeight, kernelWidth, 1],
                          strides=[1, strideHeight, strideWight, 1], padding=padding, name=name)


"""
平均值池化
"""


def avgPool(input, kernelHeight, kernelWidth, strideHeight, strideWight, name, padding="SAME"):
    return tf.nn.avg_pool(input=input, ksize=[1, kernelHeight, kernelWidth, 1],
                          strides=[1, strideHeight, strideWight, 1], padding=padding, name=name)

"""
获取Saver对象
"""
def saver():
    return tf.train.Saver()

"""
激活函数激活处理
"""
def relu(inputData,name):
    return tf.nn.relu(inputData, name=name)

"""
res残差单元
"""
def resUnit(inputData,outChannel,i=0):
    with tf.variable_scope("resUnit_"+str(i)):
        outputData=batchNorm(inputData)
        outputData=relu(inputData=outputData,name="relu")
        outputData=conv(layerName="conv1",inputData=outputData,outChannel=outChannel[0],kernelSize=[1,1])

        outputData=batchNorm(outputData)
        outputData=relu(inputData=outputData,name="relu")
        outputData=conv(layerName="conv2",inputData=outputData,outChannel=outChannel[1],kernelSize=[3,3])

        outputData = batchNorm(outputData)
        outputData = relu(inputData=outputData, name="relu")
        outputData = conv(layerName="conv3", inputData=outputData, outChannel=outChannel[2], kernelSize=[1, 1])

        return outputData+inputData

"""
全连接
Full Connection

"""
def fc( name, inputData, outChannel, trainable=True):
    shape = inputData.get_shape().as_list()
    if len(shape) == 4:
        size = shape[-1] * shape[-2] * shape[-3]
    else:
        size = shape[1]
    inputDataFlat = tf.reshape(inputData, shape=[-1, size])
    with tf.variable_scope(name):
        weights = tf.get_variable("weights", shape=[size, outChannel], dtype=tf.float32, trainable=trainable)
        biases = tf.get_variable("biases", shape=[outChannel], dtype=tf.float32, trainable=trainable)
        res = tf.nn.bias_add(tf.matmul(inputDataFlat, weights), biases)
        out = tf.nn.relu(res)
    return out

"""
优化器
"""
def getOptimizer(loss,learningRate):
    return tf.train.GradientDescentOptimizer(learningRate).minimize(loss)


def getAccuracy(input):
    return tf.reduce_mean(tf.cast(input,tf.float32))

def getLoss(probs,yInput):
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=probs, labels=yInput))

"""
获取文件，获取批量数据等，猫狗大战
"""

imgWidth = 224
imgHeight = 224
def getFile(fileDir):
    images = []
    temp = []
    for root, subFolders, files in os.walk(fileDir):
        for name in files:
            images.append(os.path.join(root, name))
        for name in subFolders:
            temp.append(os.path.join(root, name))

    labels = []
    for oneFolder in temp:
        nImg = len(os.listdir(oneFolder))
        letter = oneFolder.split("\\")[-1]
        if letter == "cat":
            labels = np.append(labels, nImg * [0])
        else:
            labels = np.append(labels, nImg * [1])
    temp = np.array([images, labels])
    temp = temp.transpose()
    np.random.shuffle(temp)
    imageList = list(temp[:, 0])
    labelList = list(temp[:, 1])
    labelList = [int(float(i)) for i in labelList]
    return imageList, labelList


def getBatch(imageList, labelList, imgWidth, imgHeight, batchSize, capacity):
    image = tf.cast(imageList, tf.string)
    label = tf.cast(labelList, tf.int32)

    inputQueue = tf.train.slice_input_producer([image, label])

    label = inputQueue[1]
    imageContents = tf.read_file(inputQueue[0])
    image = tf.image.decode_jpeg(imageContents, channels=3)

    image = tf.image.resize_image_with_crop_or_pad(image=image, target_height=imgHeight, target_width=imgWidth)
    image = tf.image.per_image_standardization(image)
    imageBatch, labelBatch = tf.train.batch(tensors=[image, label], batch_size=batchSize, num_threads=64,
                                            capacity=capacity)
    labelBatch = tf.reshape(labelBatch, [batchSize])
    return imageBatch, labelBatch


def onehot(labels):
    '''onehot编码'''
    nSample = len(labels)
    nClass = max(labels) + 1
    onehotLabels = np.zeros(shape=(nSample, nClass))
    onehotLabels[np.arange(nSample), labels] = 1
    return onehotLabels



























