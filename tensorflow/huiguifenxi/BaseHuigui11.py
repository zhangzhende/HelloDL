import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


"""
简单的线性回归和逻辑回归
"""


"""
简单一元线性回归
数据是批量输入的
问题：当产生的随机数数量很多时，无法拟合出结果
"""


def demo1_1():
    x_data = np.random.rand(10)  # 产生10个随机数，0-1内，返回一个list
    # print(x_data)
    y_data = x_data * 0.3 + 0.15  # 原始曲线
    weight = tf.Variable(0.5)
    bias = tf.Variable(0.0)
    y_model = weight * x_data + bias  # 一元模型
    loss = tf.pow((y_model - y_data), 2)  # 损失函数
    train_op = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

    sess = tf.Session()
    init = tf.initialize_all_variables()
    sess.run(init)
    for _ in range(200):
        sess.run(train_op)
        print(weight.eval(sess), bias.eval(sess))

    plt.plot(x_data, y_data, "ro", label="Original data")
    plt.plot(x_data, sess.run(weight) * x_data + sess.run(bias), label="Fitted line")
    plt.legend()
    plt.show()


"""
1_1的修改升级版，解决批量数据拟合失败的问题，由批量输入改为一个个的输入数据
>>>a = [1,2,3]
>>> b = [4,5,6]
>>> c = [4,5,6,7,8]
>>> zipped = zip(a,b)     # 打包为元组的列表
[(1, 4), (2, 5), (3, 6)]
>>> zip(a,c)              # 元素个数与最短的列表一致
[(1, 4), (2, 5), (3, 6)]
"""


def demo1_2():
    x_data = np.random.rand(100)  # 产生10个随机数，0-1内，返回一个list
    # print(x_data)
    y_data = x_data * 0.3 + 0.1  # 原始曲线
    weight = tf.Variable(0.5)
    bias = tf.Variable(0.0)
    x_ = tf.placeholder(tf.float32)
    y_ = tf.placeholder(tf.float32)
    y_model = weight * x_ + bias  # 一元模型
    loss = tf.pow((y_model - y_), 2)  # 损失函数
    train_op = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

    sess = tf.Session()
    init = tf.initialize_all_variables()
    sess.run(init)
    for _ in range(100):
        for (x, y) in zip(x_data, y_data):
            sess.run(train_op, feed_dict={x_: x, y_: y})
        print(weight.eval(sess), bias.eval(sess))

    plt.plot(x_data, y_data, "ro", label="Original data")
    plt.plot(x_data, sess.run(weight) * (x_data) + sess.run(bias), label="Fitted line")
    plt.legend()
    plt.show()


"""
1_2的审计修改版，
1.有个阈值，当损失函数到达threshold时停止计算
2.计算全改为tf内的函数
3.loss函数修改为平均值
"""


def demo1_3():
    threshold = 1.0e-5
    x_data = np.random.rand(100).astype(np.float32)  # 产生10个随机数，0-1内，返回一个list
    # print(x_data)
    y_data = x_data * 0.3 + 1  # 原始曲线
    weight = tf.Variable(1.)
    bias = tf.Variable(1.)
    x_ = tf.placeholder(tf.float32)
    y_ = tf.placeholder(tf.float32)
    y_model = tf.add(tf.multiply(x_, weight), bias)  # 一元模型
    loss = tf.reduce_mean(tf.pow((y_model - y_), 2))  # 损失函数,reduce_mean求平均值
    train_op = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

    sess = tf.Session()
    init = tf.initialize_all_variables()
    sess.run(init)
    flag = 1
    while (flag):
        for (x, y) in zip(x_data, y_data):
            sess.run(train_op, feed_dict={x_: x, y_: y})
        print(weight.eval(sess), bias.eval(sess))
        if sess.run(loss, feed_dict={x_: x_data, y_: y_data}) <= threshold:
            flag = 0
    plt.plot(x_data, y_data, "ro", label="Original data")
    plt.plot(x_data, sess.run(weight) * (x_data) + sess.run(bias), label="Fitted line")
    plt.legend()
    plt.show()


"""
多元线性回归分析

此为二元
"""


def demo2_1():
    threshold = 1.0e-2
    x1_data = np.random.rand(100).astype(np.float32)
    x2_data = np.random.rand(100).astype(np.float32)
    y_data = x1_data * 2 + x2_data * 3 + 1.5
    weight1 = tf.Variable(1.)
    weight2 = tf.Variable(1.)
    bias = tf.Variable(1.)
    x1_ = tf.placeholder(tf.float32)
    x2_ = tf.placeholder(tf.float32)
    y_ = tf.placeholder(tf.float32)
    y_model = tf.add(tf.add(tf.multiply(x1_, weight1), tf.multiply(x2_, weight2)), bias)
    loss = tf.reduce_mean(tf.pow((y_model - y_), 2))
    train_op = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

    sess = tf.Session()
    init = tf.initialize_all_variables()
    sess.run(init)
    flag = 1
    while (flag):
        for (x, y) in zip(zip(x1_data, x2_data), y_data):
            sess.run(train_op, feed_dict={x1_: x[0], x2_: x[1], y_: y})
        if sess.run(loss, feed_dict={x1_: x[0], x2_: x[1], y_: y}) <= threshold:
            flag = 0
    fig = plt.figure()
    ax = Axes3D(fig)
    X, Y = np.meshgrid(x1_data, x2_data)  # 该方法为从参数中返回一个坐标矩阵。
    Z = sess.run(weight1) * (X) + sess.run(weight2) * (Y) + sess.run(bias)
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=plt.cm.hot)
    ax.contourf(X, Y, Z, zdir="z", offset=-1, cmap=plt.cm.hot)
    ax.set_zlim(-1, 1)
    plt.show()


"""
矩阵乘法，
维度不同的矩阵相乘，会先统一化，然后再相乘，统一化包括重复行列
"""


def demo3_1():
    matrix1 = tf.constant([[3., 3.]])
    matrix2 = tf.constant([[3.], [3.]])
    sess = tf.Session()
    print(sess.run(tf.add(matrix1, matrix2)))


def demo3_2():
    a = tf.constant([[1, 2], [3, 4]])
    matrix2 = tf.placeholder("float", [2, 2])
    matrix1 = matrix2
    sess = tf.Session()
    b = sess.run(a)
    print(sess.run(matrix1, feed_dict={matrix2: b}))


"""
程序11-12
矩阵的形式完成回归分析

问题，GradientDescentOptimizer无法收敛！！！
换成AdamOptimizer可以正常收敛
"""


def demo4_1():
    houses = 100
    features = 2
    # 设计模型为2*x1+3*x2
    x_data = np.zeros([houses, 2])
    for house in range(houses):
        x_data[house, 0] = np.round(np.random.uniform(50., 150.))
        x_data[house, 1] = np.round(np.random.uniform(3., 7.))
    weights = np.array([[2.], [3.]])
    y_data = np.dot(x_data, weights)  # 100*2 的矩阵点乘 2*1的矩阵

    print(y_data.shape)
    x_data_ = tf.placeholder(tf.float32, [None, 2])
    y_data_ = tf.placeholder(tf.float32, [None, 1])
    weights_ = tf.Variable(np.ones([2, 1]), dtype=tf.float32)
    y_model = tf.matmul(x_data_, weights_)

    loss = tf.reduce_mean(tf.square((y_model - y_data_)))
    train_op = tf.train.AdamOptimizer(0.01).minimize(loss)
    # 用上面这个，下面这个无法收敛
    # train_op = tf.train.GradientDescentOptimizer(0.01).minimize(loss)
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)

    for _ in range(100):
        for x, y in zip(x_data, y_data):
            z1 = x.reshape(1, 2)
            print(z1)
            z2 = y.reshape(1, 1)
            print(z2)
            sess.run(train_op, feed_dict={x_data_: z1, y_data_: z2})
            print("weight", weights_.eval(sess))
            # print("yModel", y_model.eval(sess))


"""
跟上面一样的问题
"""


def demo4_2():
    houses = 100
    features = 2
    # 设计模型为2*x1+3*x2
    x_data = np.zeros([houses, 2])
    for house in range(houses):
        x_data[house, 0] = np.round(np.random.uniform(50., 150.))
        x_data[house, 1] = np.round(np.random.uniform(3., 7.))
    weights = np.array([[2.], [3.]])
    y_data = np.dot(x_data, weights)  # 100*2 的矩阵点乘 2*1的矩阵

    print(y_data.shape)
    x_data_ = tf.placeholder(tf.float32, [None, 2])
    # y_data_ = tf.placeholder(tf.float32, [None, 1])
    weights_ = tf.Variable(np.ones([2, 1]), dtype=tf.float32)
    y_model = tf.matmul(x_data_, weights_)

    loss = tf.reduce_mean(tf.square((y_model - y_data)))
    train_op = tf.train.AdamOptimizer(0.01).minimize(loss)
    # 用上面这个，下面这个无法收敛
    # train_op = tf.train.GradientDescentOptimizer(0.01).minimize(loss)
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)

    for _ in range(2000):
        sess.run(train_op, feed_dict={x_data_: x_data})
        print("weight", weights_.eval(sess))


"""
读取CSV文件
"""


def readFile(filename):
    filename_queue = tf.train.string_input_producer(filename, shuffle=False,num_epochs=None)
    reader = tf.TextLineReader()
    key, value = reader.read(filename_queue)
    record_defaults = [[1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0]]
    col1, col2, col3, col4, col5, col6, col7 = tf.decode_csv(value, record_defaults=record_defaults)
    label = tf.stack([col1, col2])
    features = tf.stack([col3, col4, col5, col6, col7])
    example_batch, label_batch = tf.train.shuffle_batch([features, label], batch_size=2, capacity=100,
                                                        min_after_dequeue=10)
    return example_batch, label_batch

"""
逻辑回归实现
"""
def demo5():
    example_batch, label_batch = readFile(["cancer.txt"])
    weight = tf.Variable(np.random.rand(5, 1).astype(np.float32))
    bias = tf.Variable(np.random.rand(2, 1).astype(np.float32))
    x_ = tf.placeholder(tf.float32, [None, 5])
    y_model = tf.matmul(x_, weight) + bias
    y = tf.placeholder(tf.float32, [2, 2])
    loss = -tf.reduce_sum(y * tf.log(y_model))
    train = tf.train.AdamOptimizer(0.1).minimize(loss)
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        sess.run(tf.local_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        flag = 1
        while (flag):
            e_val, l_val = sess.run([example_batch, label_batch])
            sess.run(train, feed_dict={x_: e_val, y: l_val})
            if sess.run(loss, feed_dict={x_: e_val, y: l_val}) <= 1:
                flag = 0
        print(sess.run(weight))


# demo1_1();
# demo1_2()
# demo1_3()
# demo2_1()
# demo3_1()
# demo3_2()
# demo4_1()
# demo4_2()
demo5()
