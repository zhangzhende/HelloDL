import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
import myProject.tensorflowDemo.CatVsDog.create_and_read_TFRecord2 as reader2
import os
from PIL import Image

"""
15-2
AlexNet初步实现猫狗预测,进行数据归一化处理
"""
X_train, y_train = reader2.get_file(filedir="./catanddog")
image_batch, label_batch = reader2.get_batch(image_list=X_train, label_list=y_train, img_width=227, img_height=227,
                                             batch_size=200, capacity=2048)


def batch_norm(inputs, is_training, is_conv_out=True, decay=0.999):
    scale = tf.Variable(tf.ones([inputs.get_shape()[-1]]))
    beta = tf.Variable(tf.zeros([inputs.get_shape()[-1]]))
    pop_mean = tf.Variable(tf.zeros([inputs.get_shape()[-1]]), trainable=False)
    pop_var = tf.Variable(tf.ones([inputs.get_shape()[-1]]), trainable=False)

    if is_training:
        if is_conv_out:
            # tf.nn.moments（x,axes,name=None）:计算x的均值和方差，
            # 如果x是一维的，并且axes=[0],那么计算整个向量的均值和方差，
            # axes=[0,1,2]，那么我们计算的就是卷积的全局标准化
            batch_mean, batch_var = tf.nn.moments(inputs, [0, 1, 2])
        else:
            batch_mean, batch_var = tf.nn.moments(inputs, [0])
        # 赋值
        train_mean = tf.assign(pop_mean, pop_mean * decay + batch_mean * (1 - decay))
        train_var = tf.assign(pop_var, pop_var * decay + batch_var * (1 - decay))
        # 各个网络层输出的结果归一化，以防止过拟合
        with tf.control_dependencies([train_mean, train_var]):
            return tf.nn.batch_normalization(x=inputs, mean=pop_mean, variance=pop_var, offset=beta, scale=scale,
                                             variance_epsilon=0.001)
    else:
        return tf.nn.batch_normalization(x=inputs, mean=pop_mean, variance=pop_var, offset=beta, scale=scale,
                                             variance_epsilon=0.001)


with tf.device("/cpu:0"):
    # 模型参数
    learning_rate = 1e-4
    training_iters = 200
    batch_size = 200
    display_step = 5
    n_classes = 2
    n_fc1 = 4096
    n_fc2 = 2048

    # 构建模型
    x = tf.placeholder(tf.float32, [None, 227, 227, 3])
    y = tf.placeholder(tf.int32, [None, n_classes])

    W_conv = {
        "conv1": tf.Variable(tf.truncated_normal([11, 11, 3, 96], stddev=0.0001)),
        "conv2": tf.Variable(tf.truncated_normal([5, 5, 96, 256], stddev=0.01)),
        "conv3": tf.Variable(tf.truncated_normal([3, 3, 256, 384], stddev=0.01)),
        "conv4": tf.Variable(tf.truncated_normal([3, 3, 384, 384], stddev=0.01)),
        "conv5": tf.Variable(tf.truncated_normal([3, 3, 384, 256], stddev=0.01)),
        "fc1": tf.Variable(tf.truncated_normal([6 * 6 * 256, n_fc1], stddev=0.1)),
        "fc2": tf.Variable(tf.truncated_normal([n_fc1, n_fc2], stddev=0.1)),
        "fc3": tf.Variable(tf.truncated_normal([n_fc2, n_classes], stddev=0.1))
    }
    b_conv = {
        "conv1": tf.Variable(tf.constant(0.0, dtype=tf.float32, shape=[96])),
        "conv2": tf.Variable(tf.constant(0.1, dtype=tf.float32, shape=[256])),
        "conv3": tf.Variable(tf.constant(0.1, dtype=tf.float32, shape=[384])),
        "conv4": tf.Variable(tf.constant(0.1, dtype=tf.float32, shape=[384])),
        "conv5": tf.Variable(tf.constant(0.1, dtype=tf.float32, shape=[256])),
        "fc1": tf.Variable(tf.constant(0.1, dtype=tf.float32, shape=[n_fc1])),
        "fc2": tf.Variable(tf.constant(0.1, dtype=tf.float32, shape=[n_fc2])),
        "fc3": tf.Variable(tf.constant(0.0, dtype=tf.float32, shape=[n_classes])),
    }
    x_image = tf.reshape(x, [-1, 227, 227, 3])

    # 卷积层1
    conv1 = tf.nn.conv2d(x_image, W_conv["conv1"], strides=[1, 4, 4, 1], padding="VALID")
    conv1 = tf.nn.bias_add(conv1, b_conv["conv1"])
    conv1 = batch_norm(conv1, True)
    conv1 = tf.nn.relu(conv1)

    # 池化层1
    pool1 = tf.nn.avg_pool(value=conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="VALID")
    # LRN层
    norm1 = tf.nn.lrn(input=pool1, depth_radius=5, bias=1.0, alpha=0.001 / 9.0, beta=0.75)

    # 卷积层2
    conv2 = tf.nn.conv2d(norm1, W_conv["conv2"], strides=[1, 1, 1, 1], padding="VALID")
    conv2 = tf.nn.bias_add(conv2, b_conv["conv2"])
    conv2 = batch_norm(conv2, True)
    conv2 = tf.nn.relu(conv2)
    # 池化层2
    pool2 = tf.nn.avg_pool(value=conv2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="VALID")
    # LRN层
    norm2 = tf.nn.lrn(input=pool2, depth_radius=5, bias=1.0, alpha=0.001 / 9.0, beta=0.75)

    # 卷积层3
    conv3 = tf.nn.conv2d(norm2, W_conv["conv3"], strides=[1, 1, 1, 1], padding="SAME")
    conv3 = tf.nn.bias_add(conv3, b_conv["conv3"])
    conv3 = batch_norm(conv3, True)
    conv3 = tf.nn.relu(conv3)

    # 卷积层4
    conv4 = tf.nn.conv2d(conv3, W_conv["conv4"], strides=[1, 1, 1, 1], padding="SAME")
    conv4 = tf.nn.bias_add(conv4, b_conv["conv4"])
    conv4 = batch_norm(conv4, True)
    conv4 = tf.nn.relu(conv4)

    # 卷积层5
    conv5 = tf.nn.conv2d(conv4, W_conv["conv5"], strides=[1, 1, 1, 1], padding="SAME")
    conv5 = tf.nn.bias_add(conv5, b_conv["conv5"])
    conv5 = batch_norm(conv5, True)
    conv5 = tf.nn.relu(conv5)

    # 池化层3
    pool3 = tf.nn.avg_pool(value=conv5, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="SAME")
    reshape = tf.reshape(pool3, [-1, 6 * 6 * 256])

    # 全连接层1
    fc1 = tf.add(tf.matmul(reshape, W_conv["fc1"]), b_conv["fc1"])
    fc1 = batch_norm(fc1, True,False)
    fc1 = tf.nn.relu(fc1)
    fc1 = tf.nn.dropout(fc1, 0.5)

    # 全连接层2
    fc2 = tf.add(tf.matmul(fc1, W_conv["fc2"]), b_conv["fc2"])
    fc2 = batch_norm(fc2, True, False)
    fc2 = tf.nn.relu(fc2)
    fc2 = tf.nn.dropout(fc2, 0.5)

    # 全连接层3，即分类层
    fc3 = tf.add(tf.matmul(fc2, W_conv["fc3"]), b_conv["fc3"])

    # 定义损失函数,把softmax和cross entropy放到一个函数里计算，就是为了提高运算速度,激活函数和交叉熵
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=fc3, labels=y))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)

    # 模型评估
    """
    tf.argmax就是返回最大的那个数值所在的下标。 
    test = np.array([[1, 2, 3], [2, 3, 4], [5, 4, 3], [8, 7, 2]])
np.argmax(test, 0)　　　＃输出：array([3, 3, 1]
np.argmax(test, 1)　　　＃输出：array([2, 2, 0, 0]
axis = 0: 
你就这么想，0是最大的范围，所有的数组都要进行比较，只是比较的是这些数组相同位置上的数：
test[0] = array([1, 2, 3])
test[1] = array([2, 3, 4])
test[2] = array([5, 4, 3])
test[3] = array([8, 7, 2])
# output   :    [3, 3, 1]  
axis = 1: 
　　等于1的时候，比较范围缩小了，只会比较每个数组内的数的大小，结果也会根据有几个数组，产生几个结果。
test[0] = array([1, 2, 3])  #2
test[1] = array([2, 3, 4])  #2
test[2] = array([5, 4, 3])  #0
test[3] = array([8, 7, 2])  #0
    """
#tf.equal(A, B)是对比这两个矩阵或者向量的相等的元素，如果是相等的那就返回True，反正返回False，返回的值的矩阵维度和A是一样的
    correct_pred = tf.equal(tf.argmax(fc3, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

init = tf.global_variables_initializer()

save_model = "./model/AlexNetModel.ckpt"


def train(opench):
    with tf.Session() as sess:
        sess.run(init)
        #指定一个文件用来保存图。 格式：tf.summary.FileWritter(path,sess.graph)
        train_writer = tf.summary.FileWriter("./log", sess.graph)  # 输出日志
        saver = tf.train.Saver()
        c = []
        start_time = time.time()
        # 线程协调器
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        step = 0
        for i in range(opench):
            step = i
            image, label = sess.run([image_batch, label_batch])
            labels = reader2.onehot(label)
            sess.run(optimizer, feed_dict={x: image, y: labels})
            loss_record = sess.run(loss, feed_dict={x: image, y: labels})
            print("now the loss is %f" % loss_record)

            c.append(loss_record)
            end_time = time.time()
            print("time:", (end_time - start_time))
            print("-----------%d opench is finished--------------------" % i)
        print("Optimization Finished")
        saver.save(sess,save_model)
        print("Model Save Finished")

        coord.request_stop()
        coord.join(threads=threads)
        plt.plot(c)
        plt.xlabel("Iter")
        plt.ylabel("loss")
        plt.title("lr=%f ,ti=%d , bs=%d" % (learning_rate, training_iters, batch_size))
        plt.tight_layout()
        plt.savefig("./image/cat_and_dog_AlexNet.jpg", dpi=200)


def predict_class(imageFile):
    image = Image.open(imageFile)
    image = image.resize([227, 227])
    image_array = np.array(image)

    image = tf.cast(image_array, tf.float32)
    image = tf.image.per_image_standardization(image)
    image = tf.reshape(tensor=image, shape=[1, 227, 227, 3])
    saver = tf.train.Saver()

    with tf.Session() as sess:
        save_model = tf.train.latest_checkpoint("./model")
        saver.restore(sess, save_model)
        image = tf.reshape(tensor=image, shape=[1, 227, 227, 3])
        image = sess.run(image)
        prediction = sess.run(fc3, feed_dict={x: image})
        max_index = np.argmax(prediction)
        if max_index == 0:
            return "cat"
        else:
            return "dog"
