from MyDeepLearning.Common.datasets import flowers
# from myProject.tensorflowDemo.slimAdvance import configure
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.python.slim.data.dataset_data_provider as provider

"""
Slim的简单使用
"""

flowersDataPath="../../MyDeepLearning/data/flowers/"
def parseTfTodataset():
    """
    将tfrecors文件的数据读出转化为slim需要的datasets
    
    :return: 
    """
    dataset =flowers.get_split("train",flowersDataPath)
    dataProvider=provider.DatasetDataProvider(dataset=dataset,common_queue_capacity=32,common_queue_min=1)
    image ,label=dataProvider.get(["image","label"])
    return image,label

"""
20-2
"""
def showFlowers():
    """
    展示数据集中的图片
    :return: 
    """
    with tf.Graph().as_default():
        dataset=flowers.get_split("train",flowersDataPath)
        dataProvider = provider.DatasetDataProvider(dataset=dataset, common_queue_capacity=32, common_queue_min=1)
        image, label = dataProvider.get(["image", "label"])
        with tf.Session() as sess:
            with slim.queues.QueueRunners(sess):
                for i in range(5):
                    npImage,npLabel=sess.run([image,label])
                    height,width,channel=npImage.shape
                    className=name=dataset.labels_to_names[npLabel]
                    plt.figure()
                    plt.imshow(npImage)
                    plt.title("%s,%d x %d"%(name,height,width))
                    plt.axis("off")
                    plt.show()

# showFlowers()

"""
20-3
"""
class Slim_cnn:
    """
    用slim简单完成一个CNN(卷积神经网络)模型
    """
    def __init__(self,images,numClass):
        self.images=images
        self.numClass=numClass
        self.net=self.model()

    def model(self):
        with slim.arg_scope([slim.max_pool2d],
                            kernel_size=[2,2],
                            stride=2):
            net=slim.conv2d(inputs=self.images,num_outputs=32,kernel_size=[3,3])
            net=slim.max_pool2d(inputs=net)
            net=slim.conv2d(inputs=net,num_outputs=64,kernel_size=[3,3])
            net=slim.max_pool2d(net)
            """
            TODO:flatten,扁平化最里面一层
            test=([[[1,2,3],[4,5,6],[7,8,9]],[[10,11,12],[13,14,15],[16,17,27]],[[18,19,20],[21,22,23],[24,25,26]]])    #shape is (3,3,3)
            test=slim.fatten(test)
            test.eval()
            array([[ 1,  2,  3, ...,  7,  8,  9],
                [10, 11, 12, ..., 16, 17, 27],
                [18, 19, 20, ..., 24, 25, 26]], dtype=int32)
            test.get_shape()
            TensorShape([Dimension(3), Dimension(9)])  #(3,9)"""
            net=slim.flatten(net)
            net=slim.fully_connected(net,128)
            net=slim.fully_connected(net,self.numClass,activation_fn=None)
            return net

"""
20-4
"""
def showData():
    """
    测试能否产生一个合格的数据
    :return: 
    """
    with tf.Graph().as_default():
        image=tf.random_normal([1,217,217,3])
        prob=Slim_cnn(images=image,numClass=5)
        prob=tf.nn.softmax(prob.net)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            res=sess.run(prob)
            print("res shape:")
            print(res.shape)
            print("\n res")
            print(res)

# showData()

def loadBatch(dataset,batchSize=32,height=217,width=217,isTraining=True):
    """
    从dataset中取出图片数据并进行重构，重新按批量生成数据集
    :param dataset: 
    :param batchSize: 
    :param height: 
    :param width: 
    :param isTraining: 
    :return: 
    """
    dataProvider=provider.DatasetDataProvider(dataset=dataset,common_queue_capacity=32,common_queue_min=1)
    imageRaw,label=dataProvider.get(["image","label"])
    imageRaw=tf.image.resize_images(images=imageRaw,size=[height,width])
    imageRaw=tf.image.convert_image_dtype(image=imageRaw,dtype=tf.float32)

    imageRaw,labels=tf.train.batch(
        tensors=[imageRaw,label],
        batch_size=batchSize,
        num_threads=1,
        capacity=2*batchSize
    )
    return imageRaw,labels


def trainModel():
    tf.logging.set_verbosity(tf.logging.INFO)

    dataset=flowers.get_split("train",flowersDataPath)
    images,labels=loadBatch(dataset=dataset)

    prob=Slim_cnn(images=images,numClass=5)
    prob=tf.nn.softmax(prob.net)

    oneHotLabel=slim.one_hot_encoding(labels=labels,num_classes=5)
    slim.losses.softmax_cross_entropy(prob,onehot_labels=oneHotLabel)
    totalLoss=slim.losses.get_total_loss()

    optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.01)
    trainOp=slim.learning.create_train_op(total_loss=totalLoss,optimizer=optimizer)

    finalLoss=slim.learning.train(train_op=trainOp,logdir=configure.saveModel,number_of_steps=100)

trainModel()