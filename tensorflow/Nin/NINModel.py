import tensorflow as tf

"""
17-1
NIN模型定义【network in network】

特点；首次提出了使用[1,1]的卷积核去获取跨信道信息，使用【1,1】的卷积核的网络还能实现卷积核通道数的降维和升维
"""


class NINModel:
    def __init__(self, imgs):
        self.imgs = imgs
        self.convlayers()
        self.NINResult = self.result

    def saver(self):
        return tf.train.Saver()
    #最大值池化层
    def maxPool(self, name, inputData, kernelHeight, kernelWidth, strideHeight, strideWidth):
        print(inputData.get_shape())
        out = tf.nn.max_pool(value=inputData, ksize=[1, kernelHeight, kernelWidth, 1],
                             strides=[1, strideHeight, strideWidth, 1], padding="SAME", name=name)
        return out
    #平均值池化层
    def avgPool(self, name, inputData, kernelHeight, kernelWidth, strideHeight, stridewidth):
        print(inputData.get_shape())
        return tf.nn.avg_pool(value=inputData, ksize=[1, kernelHeight, kernelWidth, 1],
                              strides=[1, strideHeight, stridewidth, 1], padding="SAME", name=name)
    #卷积层
    def conv(self, name, inputData, outChannel, kernelHeight, kernelWidth, strideHeight, strideWidth, padding="SAME"):
        print(inputData.get_shape())
        inChannel = inputData.get_shape()[-1]
        with tf.variable_scope(name):
            kernel = tf.get_variable(name="weights", shape=[kernelHeight, kernelWidth, inChannel, outChannel],
                                     dtype=tf.float32)
            biases = tf.get_variable(name="biases", shape=[outChannel], dtype=tf.float32)
            covRes = tf.nn.conv2d(input=inputData, filter=kernel, strides=[1, strideHeight, strideWidth, 1],
                                  padding=padding)
            res = tf.nn.bias_add(covRes, biases)
            out = tf.nn.relu(res)
        return out
    # relu激活函数
    def relu(self, name, inputData):
        out = tf.nn.relu(inputData, name=name)
        return out
    # 归一化
    def batch_norm(self, inputs, is_training=True, is_conv_out=True, decay=0.999):

        scale = tf.Variable(tf.ones([inputs.get_shape()[-1]]))
        beta = tf.Variable(tf.zeros([inputs.get_shape()[-1]]))
        pop_mean = tf.Variable(tf.zeros([inputs.get_shape()[-1]]), trainable=False)
        pop_var = tf.Variable(tf.ones([inputs.get_shape()[-1]]), trainable=False)

        if is_training:
            if is_conv_out:
                batch_mean, batch_var = tf.nn.moments(inputs, [0, 1, 2])
            else:
                batch_mean, batch_var = tf.nn.moments(inputs, [0])

            train_mean = tf.assign(pop_mean, pop_mean * decay + batch_mean * (1 - decay))
            train_var = tf.assign(pop_var, pop_var * decay + batch_var * (1 - decay))
            with tf.control_dependencies([train_mean, train_var]):
                return tf.nn.batch_normalization(inputs,
                                                 batch_mean, batch_var, beta, scale, 0.001)
        else:
            return tf.nn.batch_normalization(inputs,
                                             pop_mean, pop_var, beta, scale, 0.001)

    def convlayers(self):
        # conv1
        self.outData = self.conv(name="conv1", inputData=self.imgs, outChannel=48, kernelHeight=7, kernelWidth=7,
                                 strideHeight=4, strideWidth=4)  # (?,224,224,3)
        self.outData = self.batch_norm(self.outData)
        self.outData = self.relu(name="relu", inputData=self.outData)
        self.outData = self.conv(name="cccp1", inputData=self.outData, outChannel=48, kernelHeight=1, kernelWidth=1,
                                 strideHeight=1, strideWidth=1)  # (?,56,56,96)
        self.outData = self.relu(name="relu", inputData=self.outData)
        self.outData = self.conv(name="cccp2", inputData=self.outData, outChannel=48, kernelHeight=1, kernelWidth=1,
                                 strideHeight=1, strideWidth=1)  # (?,56,56,96)
        self.outData = self.maxPool(name="pool1", inputData=self.outData, kernelHeight=2, kernelWidth=2, strideHeight=2,
                                    strideWidth=2)  # (?,56,56,96)

        # conv2
        self.outData = self.conv(name="conv2", inputData=self.outData, outChannel=72, kernelHeight=5, kernelWidth=5,
                                 strideHeight=1, strideWidth=1)  # (?,28,28,96)
        self.outData = self.batch_norm(inputs=self.outData)
        self.outData = self.relu(name="relu", inputData=self.outData)
        self.outData = self.conv(name="cccp3", inputData=self.outData, outChannel=72, kernelHeight=1, kernelWidth=1,
                                 strideHeight=1, strideWidth=1)  # (?,28,28,256)
        self.outData = self.relu(name="relu", inputData=self.outData)
        self.outData = self.conv(name="cccp4", inputData=self.outData, outChannel=72, kernelHeight=1, kernelWidth=1,
                                 strideHeight=1, strideWidth=1)  # (?,28,28,256)
        self.outData = self.maxPool(name="pool2", inputData=self.outData, kernelHeight=2, kernelWidth=2, strideHeight=2,
                                    strideWidth=2)  # (?,28,28,256)

        # conv3
        self.outData = self.conv(name="conv3", inputData=self.outData, outChannel=32, kernelHeight=3, kernelWidth=3,
                                 strideHeight=1, strideWidth=1)  # (?,14,14,256)
        self.outData = self.batch_norm(inputs=self.outData)
        self.outData = self.relu(name="relu", inputData=self.outData)
        self.outData = self.conv(name="cccp5", inputData=self.outData, outChannel=32, kernelHeight=1, kernelWidth=1,
                                 strideHeight=1, strideWidth=1)  # (?,14,14,128)
        self.outData = self.relu(name="relu", inputData=self.outData)
        self.outData = self.conv(name="cccp6", inputData=self.outData, outChannel=32, kernelHeight=1, kernelWidth=1,
                                 strideHeight=1, strideWidth=1)  # (?,14,14,128)
        self.outData = self.maxPool(name="pool3", inputData=self.outData, kernelHeight=2, kernelWidth=2, strideHeight=2,
                                    strideWidth=2)  # (?,14,14,128)

        # conv4
        self.outData = self.conv(name="conv4", inputData=self.outData, outChannel=1024, kernelHeight=5, kernelWidth=5,
                                 strideHeight=1, strideWidth=1)  # (?,7,7,128)
        self.outData = self.batch_norm(inputs=self.outData)
        self.outData = self.relu(name="relu", inputData=self.outData)
        self.outData = self.conv(name="cccp7", inputData=self.outData, outChannel=1000, kernelHeight=1, kernelWidth=1,
                                 strideHeight=1, strideWidth=1)  # (?,7,7,2)
        self.outData = self.relu(name="relu", inputData=self.outData)
        self.outData = self.conv(name="cccp8", inputData=self.outData, outChannel=2, kernelHeight=1, kernelWidth=1,
                                 strideHeight=2, strideWidth=2)  # (?,7,7,2)

        print("here shape is :", self.outData.get_shape())
        self.outData = self.avgPool(name="avgpool", inputData=self.outData, kernelHeight=7, kernelWidth=7,
                                    strideHeight=4, stridewidth=4)
        print("here shape is :", self.outData.get_shape())
        self.result = tf.reshape(self.outData, shape=[-1, 2])
        print("here is model_2 and result shape is :", self.result.get_shape())
