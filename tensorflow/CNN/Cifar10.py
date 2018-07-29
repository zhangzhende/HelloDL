import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pickle as pickle
import time

"""
14-
简单使用卷积神经网络实现cifar10图像数据分类
"""


"""
加载文件数据
"""


def unpickle(filename):
    with open(filename, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

"""
设置标签
"""
def onehot(labels):
    """one hot 编码"""
    n_sample = len(labels)  # 标签数量
    n_class = max(labels) + 1  # 标签种类
    onehot_labels = np.zeros((n_sample, n_class))
    onehot_labels[np.arange(n_sample), labels] = 1
    return onehot_labels

"""
输出电脑CPU,GPU配置信息
"""
def ComputerInfo():
    from tensorflow.python.client import device_lib
    print(device_lib.list_local_devices())
"""
CIFAR-10模型构建与数据处理
图片分类实例
训练很慢，但是没有出错
"""


def main():
    """训练集数据"""
    data1 = unpickle("./cifar10/batches_py/data_batch_1")
    data2 = unpickle("./cifar10/batches_py/data_batch_2")
    data3 = unpickle("./cifar10/batches_py/data_batch_3")
    data4 = unpickle("./cifar10/batches_py/data_batch_4")
    data5 = unpickle("./cifar10/batches_py/data_batch_5")
    print(data1.keys())
    # TODO
    #concatenate(x,axis=0) 这个函数用于将多个数组进行连接，这与stack函数很容易混淆，
    # 他们之间的区别是concatenate会把当前要匹配的元素降一维，即去掉最外面那一层括号
    X_train = np.concatenate((data1[b'data'], data2[b'data'], data3[b'data'], data4[b'data'], data5[b'data']), axis=0)
    Y_train = np.concatenate((data1[b'labels'], data2[b'labels'], data3[b'labels'], data4[b'labels'], data5[b'labels']),
                             axis=0)
    Y_train = onehot(Y_train)
    """测试集数据"""
    test = unpickle("./cifar10/batches_py/test_batch")
    X_test = test[b"data"][:5000, :]
    Y_test = onehot(test[b"labels"])[:5000, :]

    print("Training dataset shape:", X_train.shape)
    print("Training labels shape:", Y_train.shape)
    print("Testing dataset shape:", X_test.shape)
    print("Testing labels shape:", Y_test.shape)
    # TODO
    #指定使用的CPU
    with tf.device("/cpu:0"):

        # 参数
        learing_rate = 1e-3
        training_iters = 200
        batch_size = 50
        display_step = 5
        n_features = 3072
        n_classes = 10
        n_fc1 = 384
        n_fc2 = 192

        # 构建模型
        x = tf.placeholder(tf.float32, [None, n_features])
        y = tf.placeholder(tf.float32, [None, n_classes])

        W_conv = {
            # TODO
            #tf.truncated_normal(shape, mean=0.0, stddev=1.0, dtype=tf.float32, seed=None, name=None)
            #从截断的正态分布中输出随机值。
            #生成的值服从具有指定平均值和标准偏差的正态分布，如果生成的值大于平均值2个标准偏差的值则丢弃重新选择。
            """tf.random_normal(shape, mean=0.0, stddev=1.0, dtype=tf.float32, seed=None, name=None)
            从正态分布中输出随机值。
            参数:
            shape: 一维的张量，也是输出的张量。
            mean: 正态分布的均值。
            stddev: 正态分布的标准差。
            dtype: 输出的类型。
            seed: 一个整数，当设置之后，每次生成的随机数都一样。
            name: 操作的名字。"""
            "conv1": tf.Variable(tf.truncated_normal([5, 5, 3, 32], stddev=0.0001)),
            "conv2": tf.Variable(tf.truncated_normal([5, 5, 32, 64], stddev=0.01)),
            "fc1": tf.Variable(tf.truncated_normal([8 * 8 * 64, n_fc1], stddev=0.1)),
            "fc2": tf.Variable(tf.truncated_normal([n_fc1, n_fc2], stddev=0.1)),
            "fc3": tf.Variable(tf.truncated_normal([n_fc2, n_classes], stddev=0.1)),
        }
        b_conv = {
            "conv1": tf.Variable(tf.constant(0.0, dtype=tf.float32, shape=[32])),
            "conv2": tf.Variable(tf.constant(0.1, dtype=tf.float32, shape=[64])),
            "fc1": tf.Variable(tf.constant(0.1, dtype=tf.float32, shape=[n_fc1])),
            "fc2": tf.Variable(tf.constant(0.1, dtype=tf.float32, shape=[n_fc2])),
            "fc3": tf.Variable(tf.constant(0.0, dtype=tf.float32, shape=[n_classes])),
        }
        x_image = tf.reshape(x, [-1, 32, 32, 3])
        # 卷积层1
        conv1 = tf.nn.conv2d(x_image, W_conv["conv1"], strides=[1, 1, 1, 1], padding="SAME")
        conv1 = tf.nn.bias_add(conv1, b_conv["conv1"])
        conv1 = tf.nn.relu(conv1)
        # 池化层1
        pool1 = tf.nn.avg_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="SAME")
        # LRN层
        #TODO
        #LRN函数类似DROPOUT和数据增强作为relu激励之后防止数据过拟合而提出的一种处理方法,全称是 local response normalization--局部响应标准化。
        norm1 = tf.nn.lrn(pool1, depth_radius=4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
        # 卷积层2
        conv2 = tf.nn.conv2d(input=norm1, filter=W_conv["conv2"], strides=[1, 1, 1, 1], padding="SAME")
        conv2 = tf.nn.bias_add(conv2, b_conv["conv2"])
        conv2 = tf.nn.relu(conv2)
        # LRN层
        norm2 = tf.nn.lrn(input=conv2, depth_radius=4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
        # 池化层
        pool2 = tf.nn.avg_pool(value=norm2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="SAME")
        reshape = tf.reshape(pool2, [-1, 8 * 8 * 64])

        fc1 = tf.add(tf.matmul(reshape, W_conv["fc1"]), b_conv["fc1"])
        fc1 = tf.nn.relu(fc1)
        # 全连接层2
        fc2 = tf.add(tf.matmul(fc1, W_conv["fc2"]), b_conv["fc2"])
        fc2 = tf.nn.relu(fc2)
        # 全连接层3
        fc3 = tf.nn.softmax(tf.add(tf.matmul(fc2, W_conv["fc3"]), b_conv["fc3"]))
        # 损失函数
        #TODO
        #softmax_cross_entropy_with_logits::总之，tensorflow之所以把softmax和cross entropy放到一个函数里计算，就是为了提高运算速度
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=fc3, labels=y))
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=learing_rate).minimize(loss)
        # 评估moxing
        correct_pred = tf.equal(tf.argmax(fc3, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        init = tf.global_variables_initializer()

        with tf.Session() as sess:
            sess.run(init)
            c = []
            total_batch = int(X_train.shape[0] / batch_size)  # 要跑多少波
            start_time = time.time()
            for i in range(200):
                for batch in range(total_batch):
                    batch_x = X_train[batch * batch_size:(batch + 1) * batch_size, :]
                    batch_y = Y_train[batch * batch_size:(batch + 1) * batch_size, :]
                    sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
                acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y})
                print(acc)
                c.append(acc)
                end_time = time.time()
                print("time:", (end_time - start_time))
                start_time = end_time
                print("-" * 10, "%d onpech is finished" % (i), "-" * 10)
            print("Optimization Finished!!")

            # Test
            test_acc = sess.run(accuracy, feed_dict={x: X_test, y: Y_test})
            print("Testing accuracy:", test_acc)
            plt.plot(c)
            plt.xlabel("Iter")
            plt.ylabel("Cost")
            plt.title("ir=%f,ti=%d,bs=%d,acc=%f" % (learing_rate, training_iters, batch_size, test_acc))
            plt.tight_layout()
            plt.savefig("cnn-tf-cifar10-%s.png" % test_acc, dpi=200)


# main()
ComputerInfo()