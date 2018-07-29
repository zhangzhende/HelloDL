import tensorflow as tf
import numpy as np
import os

imgWidth = 224
imgHeight = 224
"""
16-14

"""

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
