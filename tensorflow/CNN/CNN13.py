import struct
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import cv2
from tensorflow.examples.tutorials.mnist import input_data
import time

"""
13-
初步了解卷积神经网络CNN
"""


"""
13-1
手动实现卷积操作
"""


def convolve(dataMat, kernel):
    m, n = dataMat.shape
    km, kn = kernel.shape
    newMat = np.ones(((m - km + 1), (n - kn + 1)))
    tempMat = np.ones(((km), (kn)))

    for row in range(m - km + 1):
        for col in range(n - kn + 1):
            for m_k in range(km):
                for n_k in range(kn):
                    tempMat[m_k, n_k] = dataMat[(row + m_k), (col + n_k)] * kernel[m_k, n_k]
        newMat[row, col] = np.sum(tempMat)
    return newMat


"""
13-2
使用卷积简单处理矩阵
3*3的图片，输出1通道
1*1的卷积核
"""


def demo1():
    input = tf.Variable(tf.random_normal([1, 3, 3, 1]))
    filter = tf.Variable(tf.ones([1, 1, 1, 1]))

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        conv2d = tf.nn.conv2d(input=input, filter=filter, strides=[1, 1, 1, 1], padding="VALID")
        print(sess.run(conv2d))


"""
13-3
5*5的图片，5通道
3*3的卷积核，不含边缘
"""


def demo2():
    input = tf.Variable(tf.random_normal([1, 5, 5, 5]))
    filter = tf.Variable(tf.ones([3, 3, 5, 1]))

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        conv2d = tf.nn.conv2d(input=input, filter=filter, strides=[1, 1, 1, 1], padding="VALID")
        print(sess.run(conv2d))


"""
13-4
5*5的图片，5通道
3*3的卷积核，含边缘,外圈补0计算
"""


def demo3():
    input = tf.Variable(tf.random_normal([1, 5, 5, 5]))
    filter = tf.Variable(tf.ones([3, 3, 5, 1]))

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        conv2d = tf.nn.conv2d(input=input, filter=filter, strides=[1, 1, 1, 1], padding="SAME")
        print(sess.run(conv2d))


"""
图片卷积操作
512*512的图片，2*2的卷积核
"""


def demo4():
    img = cv2.imread("./jpg/001/1.jpg")
    img = cv2.resize(img, (512, 512))
    img = np.array(img, dtype=np.float32)
    x_image = tf.reshape(img, [1, 512, 512, 3])

    filter = tf.Variable(tf.ones([2, 2, 3, 1]))

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        res = tf.nn.conv2d(x_image, filter=filter, strides=[1, 1, 1, 1], padding="SAME")
        res_image = sess.run(tf.reshape(res, [512, 512])) / 12 + 1
    cv2.imshow("hello", res_image.astype("uint8"))
    cv2.waitKey()


"""
图片卷积操作
512*512的图片，7*7的卷积核
"""


def demo5():
    img = cv2.imread("./jpg/001/1.jpg")
    img = cv2.resize(img, (512, 512))
    img = np.array(img, dtype=np.float32)
    x_image = tf.reshape(img, [1, 512, 512, 3])

    filter = tf.Variable(tf.ones([7, 7, 3, 1]))

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        res = tf.nn.conv2d(x_image, filter=filter, strides=[1, 1, 1, 1], padding="SAME")
        res_image = sess.run(tf.reshape(res, [512, 512])) / 128 + 1
    cv2.imshow("hello", res_image.astype("uint8"))
    cv2.waitKey()


"""
13-8
"""


def demo6():
    data = tf.constant([[
        [3.0, 2.0, 3.0, 4.0],
        [2.0, 6.0, 2.0, 4.0],
        [1.0, 2.0, 1.0, 5.0],
        [4.0, 3.0, 2.0, 1.0]
    ]])
    data = tf.reshape(data, [1, 4, 4, 1])
    maxpooling = tf.nn.max_pool(data, [1, 2, 2, 1], [1, 2, 2, 1], padding="VALID")

    with tf.Session() as sess:
        print(sess.run(maxpooling))


"""
13-9
z在卷积以后加入一步池化操作

"""


def demo7():
    img = cv2.imread("./jpg/001/1.jpg")
    img = cv2.resize(img, (512, 512))
    img = np.array(img, dtype=np.float32)
    x_image = tf.reshape(img, [1, 512, 512, 3])

    filter = tf.Variable(tf.ones([7, 7, 3, 1]))

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        res = tf.nn.conv2d(x_image, filter=filter, strides=[1, 1, 1, 1], padding="SAME")
        res = tf.nn.max_pool(res, [1, 2, 2, 1], [1, 2, 2, 1], padding="VALID")
        res_image = sess.run(tf.reshape(res, [256, 256])) / 150 + 1
    cv2.imshow("hello", res_image.astype("uint8"))
    cv2.waitKey()


"""
13-10
卷积神经网络LeNet 实现
完成手写体识别
"""


def demo8():
    x = tf.placeholder(tf.float32, [None, 784])
    y_ = tf.placeholder(tf.float32, [None, 10])

    x_image = tf.reshape(x, [-1, 28, 28, 1])

    # 第一个卷积层
    filter1 = tf.Variable(tf.truncated_normal([5, 5, 1, 6]))  # 卷积核，5*5,1通道输入，6通道输出,6个不同的卷积核
    bias1 = tf.Variable(tf.truncated_normal([6]))
    conv1 = tf.nn.conv2d(x_image, filter=filter1, strides=[1, 1, 1, 1], padding="SAME")
    h_conv1 = tf.nn.sigmoid(conv1 + bias1)
    # 第一个池化层
    maxpool2 = tf.nn.max_pool(h_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
    # 第二个卷积层
    filter2 = tf.Variable(tf.truncated_normal([5, 5, 6, 16]))
    bias2 = tf.Variable(tf.truncated_normal([16]))
    conv2 = tf.nn.conv2d(maxpool2, filter=filter2, strides=[1, 1, 1, 1], padding="SAME")
    h_conv2 = tf.nn.sigmoid(conv2 + bias2)
    # 第二个池化层
    maxpool3 = tf.nn.max_pool(h_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
    # 第三个卷积层
    filter3 = tf.Variable(tf.truncated_normal([5, 5, 16, 120]))
    bias3 = tf.Variable(tf.truncated_normal([120]))
    conv3 = tf.nn.conv2d(maxpool3, filter=filter3, strides=[1, 1, 1, 1], padding="SAME")
    h_conv3 = tf.nn.sigmoid(conv3 + bias3)
    # 全连接层
    W_fc1 = tf.Variable(tf.truncated_normal([7 * 7 * 120, 80]))  # 权值参数
    b_fc1 = tf.Variable(tf.truncated_normal([80]))  # 偏置值
    h_pool2_flat = tf.reshape(h_conv3, [-1, 7 * 7 * 120])
    h_fc1 = tf.nn.sigmoid(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
    # 输出层，使用softmax进行多分类
    W_fc2 = tf.Variable(tf.truncated_normal([80, 10]))
    b_fc2 = tf.Variable(tf.truncated_normal([10]))
    y_conv = tf.nn.softmax(tf.matmul(h_fc1, W_fc2) + b_fc2)
    # 损失函数
    cross_entropy = -tf.reduce_sum(y_ * tf.log(y_conv))
    # 使用GDO爱优化算法调整参数
    train_step = tf.train.GradientDescentOptimizer(0.001).minimize(cross_entropy)
    sess = tf.InteractiveSession()
    # 测试正确率
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    # 初始化所有变量
    sess.run(tf.initialize_all_variables())
    # 获取数据集
    mnist_data_set = input_data.read_data_sets("MNIST_data", one_hot=True)

    start_time = time.time()
    for i in range(20000):
        # 获取训练数据
        batch_xs, batch_ys = mnist_data_set.train.next_batch(200)

        if i % 2 == 0:
            train_accuracy = accuracy.eval(feed_dict={x: batch_xs, y_: batch_ys})
            print("step %d,training accuracy %g" % (i, train_accuracy))
            end_time = time.time()
            print("time:", (end_time - start_time))
            start_time = end_time
        train_step.run(feed_dict={x: batch_xs, y_: batch_ys})
    sess.close()


"""
13-11
卷积神经网络LeNet 实现
完成手写体识别
相对于上一个，将激活函数换成了relu
但是崩了，并不能收敛准确率，正确率一直在0.1左右跳
"""


def demo9():
    x = tf.placeholder(tf.float32, [None, 784])
    y_ = tf.placeholder(tf.float32, [None, 10])

    x_image = tf.reshape(x, [-1, 28, 28, 1])

    # 第一个卷积层
    filter1 = tf.Variable(tf.truncated_normal([5, 5, 1, 6]))  # 卷积核，5*5,1通道输入，6通道输出,6个不同的卷积核
    bias1 = tf.Variable(tf.truncated_normal([6]))
    conv1 = tf.nn.conv2d(x_image, filter=filter1, strides=[1, 1, 1, 1], padding="SAME")
    h_conv1 = tf.nn.relu(conv1 + bias1)
    # 第一个池化层
    maxpool2 = tf.nn.max_pool(h_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
    # 第二个卷积层
    filter2 = tf.Variable(tf.truncated_normal([5, 5, 6, 16]))
    bias2 = tf.Variable(tf.truncated_normal([16]))
    conv2 = tf.nn.conv2d(maxpool2, filter=filter2, strides=[1, 1, 1, 1], padding="SAME")
    h_conv2 = tf.nn.relu(conv2 + bias2)
    # 第二个池化层
    maxpool3 = tf.nn.max_pool(h_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
    # 第三个卷积层
    filter3 = tf.Variable(tf.truncated_normal([5, 5, 16, 120]))
    bias3 = tf.Variable(tf.truncated_normal([120]))
    conv3 = tf.nn.conv2d(maxpool3, filter=filter3, strides=[1, 1, 1, 1], padding="SAME")
    h_conv3 = tf.nn.relu(conv3 + bias3)
    # 全连接层
    W_fc1 = tf.Variable(tf.truncated_normal([7 * 7 * 120, 80]))  # 权值参数
    b_fc1 = tf.Variable(tf.truncated_normal([80]))  # 偏置值
    h_pool2_flat = tf.reshape(h_conv3, [-1, 7 * 7 * 120])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
    # 输出层，使用softmax进行多分类
    W_fc2 = tf.Variable(tf.truncated_normal([80, 10]))
    b_fc2 = tf.Variable(tf.truncated_normal([10]))
    y_conv = tf.nn.softmax(tf.matmul(h_fc1, W_fc2) + b_fc2)
    # 损失函数
    cross_entropy = -tf.reduce_sum(y_ * tf.log(y_conv))
    # 使用GDO爱优化算法调整参数
    # train_step = tf.train.GradientDescentOptimizer(0.0001).minimize(cross_entropy)
    train_step = tf.train.AdamOptimizer(0.0001).minimize(cross_entropy)
    sess = tf.InteractiveSession()
    # 测试正确率
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    # 初始化所有变量
    sess.run(tf.initialize_all_variables())
    # 获取数据集
    mnist_data_set = input_data.read_data_sets("MNIST_data", one_hot=True)

    start_time = time.time()
    for i in range(20000):
        # 获取训练数据
        batch_xs, batch_ys = mnist_data_set.train.next_batch(200)

        if i % 2 == 0:
            train_accuracy = accuracy.eval(feed_dict={x: batch_xs, y_: batch_ys})
            print("step %d,training accuracy %g" % (i, train_accuracy))
            end_time = time.time()
            print("time:", (end_time - start_time))
            start_time = end_time
        train_step.run(feed_dict={x: batch_xs, y_: batch_ys})
    sess.close()

"""
13-12
代码重构，模块化
"""
#初始化权值值
def weight_variable(shape):
    initial=tf.truncated_normal(shape=shape,stddev=0.1)#随机生成一个标准差为0.1的矩阵
    return tf.Variable(initial)
#初始化单个卷积核上的偏置值
def bias_variable(shape):
    initial=tf.constant(0.1,shape=shape)
    return tf.Variable(initial)
#输入矩阵x，用卷积核W进行卷积运算，strides表示步长
def conv2d(x,W):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding="SAME")
#对X进行最大池化操作
def max_pool_2x2(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")

"""
13-12
重构代码后
"""
def demo10():
    sess =tf.InteractiveSession()
    x=tf.placeholder(tf.float32,[None,784])
    y_=tf.placeholder(tf.float32,[None,10])

    x_image=tf.reshape(x,[-1,28,28,1])

    W_conv1=weight_variable([5,5,1,6])
    b_conv1=bias_variable([6])
    h_conv1=tf.nn.relu(conv2d(x_image,W_conv1)+b_conv1)
    h_pool1=max_pool_2x2(h_conv1)

    W_conv2 = weight_variable([5, 5, 6, 16])
    b_conv2 = bias_variable([16])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    W_fc1=weight_variable([7*7*16,120])
    b_fc1=bias_variable([120])
    h_pool2_flat=tf.reshape(h_pool2,[-1,7*7*16])

    h_fc1=tf.nn.relu(tf.matmul(h_pool2_flat,W_fc1)+b_fc1)

    W_fc2=weight_variable([120,10])
    b_fc2=bias_variable([10])
    y_conv=tf.nn.softmax(tf.matmul(h_fc1,W_fc2)+b_fc2)

    # 损失函数
    cross_entropy = -tf.reduce_sum(y_ * tf.log(y_conv))
    # 使用GDO爱优化算法调整参数
    # train_step = tf.train.GradientDescentOptimizer(0.0001).minimize(cross_entropy)
    train_step = tf.train.AdamOptimizer(0.0001).minimize(cross_entropy)
    # 测试正确率
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    # 初始化所有变量
    sess.run(tf.initialize_all_variables())
    # 获取数据集
    mnist_data_set = input_data.read_data_sets("MNIST_data", one_hot=True)
    c=[]
    start_time = time.time()
    for i in range(1000):
        # 获取训练数据
        batch_xs, batch_ys = mnist_data_set.train.next_batch(200)

        if i % 2 == 0:
            train_accuracy = accuracy.eval(feed_dict={x: batch_xs, y_: batch_ys})
            c.append(train_accuracy)
            print("step %d,training accuracy %g" % (i, train_accuracy))
            end_time = time.time()
            print("time:", (end_time - start_time))
            start_time = end_time
        train_step.run(feed_dict={x: batch_xs, y_: batch_ys})
    sess.close()
    plt.plot(c)
    plt.tight_layout()
    plt.savefig("cnn-tf-cifar10-2",dpi=200)
"""
13-13
在上面的13-12上做了参数调整
修改了每个隐藏层中神经元的数目，即第一次生成了32个通道的卷积层，第二层为64，全连接使用1024个神经元作为学习参数
结论：计算相对变慢，更复杂了
真确率上升速度提高，同时波动范围也相对变小，更稳定
"""
def demo11():
    sess = tf.InteractiveSession()
    x = tf.placeholder(tf.float32, [None, 784])
    y_ = tf.placeholder(tf.float32, [None, 10])

    x_image = tf.reshape(x, [-1, 28, 28, 1])

    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    W_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])

    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])
    y_conv = tf.nn.softmax(tf.matmul(h_fc1, W_fc2) + b_fc2)

    # 损失函数
    cross_entropy = -tf.reduce_sum(y_ * tf.log(y_conv))
    # 使用GDO爱优化算法调整参数
    # train_step = tf.train.GradientDescentOptimizer(0.0001).minimize(cross_entropy)
    train_step = tf.train.AdamOptimizer(0.0001).minimize(cross_entropy)
    # 测试正确率
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    # 初始化所有变量
    sess.run(tf.initialize_all_variables())
    # 获取数据集
    mnist_data_set = input_data.read_data_sets("MNIST_data", one_hot=True)
    c = []
    start_time = time.time()
    for i in range(1000):
        # 获取训练数据
        batch_xs, batch_ys = mnist_data_set.train.next_batch(200)

        if i % 2 == 0:
            train_accuracy = accuracy.eval(feed_dict={x: batch_xs, y_: batch_ys})
            c.append(train_accuracy)
            print("step %d,training accuracy %g" % (i, train_accuracy))
            end_time = time.time()
            print("time:", (end_time - start_time))
            start_time = end_time
        train_step.run(feed_dict={x: batch_xs, y_: batch_ys})
    sess.close()
    plt.plot(c)
    plt.tight_layout()
    plt.savefig("cnn-tf-cifar10-1", dpi=200)

def demo12():
    sess = tf.InteractiveSession()
    x = tf.placeholder(tf.float32, [None, 784])
    y_ = tf.placeholder(tf.float32, [None, 10])

    x_image = tf.reshape(x, [-1, 28, 28, 1])

    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    W_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
    #增加一层全连接层
    W_fc2 = weight_variable([1024, 128])
    b_fc2 = bias_variable([128])
    h_fc2=tf.nn.relu(tf.matmul(h_fc1, W_fc2) + b_fc2)

    W_fc3 = weight_variable([128, 10])
    b_fc3 = bias_variable([10])
    y_conv = tf.nn.softmax(tf.matmul(h_fc2, W_fc3) + b_fc3)

    # 损失函数
    cross_entropy = -tf.reduce_sum(y_ * tf.log(y_conv))
    # 使用GDO爱优化算法调整参数
    # train_step = tf.train.GradientDescentOptimizer(0.0001).minimize(cross_entropy)
    train_step = tf.train.AdamOptimizer(0.0001).minimize(cross_entropy)
    # 测试正确率
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    # 初始化所有变量
    sess.run(tf.initialize_all_variables())
    # 获取数据集
    mnist_data_set = input_data.read_data_sets("MNIST_data", one_hot=True)
    c = []
    start_time = time.time()
    for i in range(1000):
        # 获取训练数据
        batch_xs, batch_ys = mnist_data_set.train.next_batch(200)

        if i % 2 == 0:
            train_accuracy = accuracy.eval(feed_dict={x: batch_xs, y_: batch_ys})
            c.append(train_accuracy)
            print("step %d,training accuracy %g" % (i, train_accuracy))
            end_time = time.time()
            print("time:", (end_time - start_time))
            start_time = end_time
        train_step.run(feed_dict={x: batch_xs, y_: batch_ys})
    sess.close()
    plt.plot(c)
    plt.tight_layout()
    plt.savefig("cnn-tf-cifar10-3", dpi=200)




# demo1()
# demo2()
# demo3()
# demo4()
# demo5()
# demo6()
# demo7()
# demo8()
# demo9()
# demo10()
demo11()
demo12()

