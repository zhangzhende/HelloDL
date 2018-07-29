import tensorflow as tf

"""
VGG16模型的TensorFlow实现
"""
class vgg16:
    def __init__(self, imgs):
        self.imgs = imgs
        self.convLayers()
        self.fcLayers(nClass=3)
        self.probs = self.fc8

    def saver(self):
        return tf.train.Saver()

    def maxpool(self, name, inputData):
        out = tf.nn.max_pool(inputData, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME", name=name)
        return out

    def conv(self, name, inputData, outChannel):
        inChannel = inputData.get_shape()[-1]
        with tf.VariableScope(name):
            kernel = tf.get_variable("weights", shape=[3, 3, inChannel, outChannel], dtype=tf.float32, trainable=False)
            biases = tf.get_variable("biases", shape=[outChannel], dtype=tf.float32, trainable=False)
            convRes = tf.nn.conv2d(input=inputData, filter=kernel, strides=[1, 1, 1, 1], padding="SAME")
            res = tf.nn.bias_add(convRes, bias=biases)
            out = tf.nn.relu(res, biases)
        self.parameters += [kernel, biases]
        return out

    def fc(self, name, inputData, outChannel, trainable=True):
        shape = inputData.get_shape().as_list()
        if len(shape) == 4:
            size = shape[-1] * shape[-2] * shape[-3]
        else:
            size = shape[1]
        inputDataFlat = tf.reshape(inputData, shape=[-1, size])
        with tf.variable_scope(name):
            weights = tf.get_variable("weights", shape=[size, outChannel], dtype=tf.float32, trainable=trainable)
            biases = tf.get_variable("weights", shape=[outChannel], dtype=tf.float32, trainable=trainable)
            res = tf.nn.bias_add(tf.matmul(inputDataFlat, weights), biases)
            out = tf.nn.relu(res)
        self.parameters += [weights, biases]
        return out

    def convLayers(self):
        # conv1
        self.conv1_1 = self.conv(name="conv1_1", inputData=self.imgs, outChannel=64)
        self.conv1_2 = self.conv(name="conv1_2", inputData=self.conv1_1, outChannel=64)
        self.pool1 = self.maxpool(name="pool1", inputData=self.conv1_2)

        # conv2
        self.conv2_1 = self.conv(name="conv2_1", inputData=self.pool1, outChannel=128)
        self.conv2_2 = self.conv(name="conv2_2", inputData=self.conv2_1, outChannel=128)
        self.pool2 = self.maxpool(name="pool2", inputData=self.conv2_2)
        # conv3
        self.conv3_1 = self.conv(name="conv3_1", inputData=self.pool2, outChannel=256)
        self.conv3_2 = self.conv(name="conv3_2", inputData=self.conv3_1, outChannel=256)
        self.conv3_3 = self.conv(name="conv3_3", inputData=self.conv3_2, outChannel=256)
        self.pool3 = self.maxpool(name="pool3", inputData=self.conv3_3)
        # conv4
        self.conv4_1 = self.conv(name="conv4_1", inputData=self.pool3, outChannel=512)
        self.conv4_2 = self.conv(name="conv4_2", inputData=self.conv4_1, outChannel=512)
        self.conv4_3 = self.conv(name="conv4_3", inputData=self.conv4_2, outChannel=512)
        self.pool4 = self.maxpool(name="pool4", inputData=self.conv4_3)
        # conv5
        self.conv5_1 = self.conv(name="conv5_1", inputData=self.pool4, outChannel=512)
        self.conv5_2 = self.conv(name="conv5_2", inputData=self.conv5_1, outChannel=512)
        self.conv5_3 = self.conv(name="conv5_3", inputData=self.conv5_2, outChannel=512)
        self.pool5 = self.maxpool(name="pool5", inputData=self.conv5_3)

    def fcLayers(self, nClass):
        self.fc6 = self.fc(name="fc6", inputData=self.pool5, outChannel=4096, trainable=False)
        self.fc7 = self.fc(name="fc7", inputData=self.fc6, outChannel=4096, trainable=False)
        self.fc8 = self.fc(name="fc8", inputData=self.fc7, outChannel=nClass)
