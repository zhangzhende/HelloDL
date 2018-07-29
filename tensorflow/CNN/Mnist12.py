import tensorflow.examples.tutorials.mnist.input_data as input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
import tensorflow as tf


"""
深度学习的hello world!! 
"""
"""
最简单的手写体识别
损失函数为最小二乘法
"""
def demo1():
    x_data = tf.placeholder(tf.float32, [None, 784])
    weight = tf.Variable(tf.ones([784, 10]))
    bias = tf.Variable(tf.ones([10]))
    y_model = tf.nn.softmax(tf.matmul(x_data, weight) + bias)
    y_data = tf.placeholder(tf.float32, [None, 10])
    loss = tf.reduce_sum(tf.square(y_model - y_data))
    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(loss)
    init = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init)
    for _ in range(1000):
        batch_xs,batch_ys=mnist.train.next_batch(100)
        sess.run(train_step,feed_dict={x_data:batch_xs,y_data:batch_ys})
        if _%50==0:
            correct_prediction=tf.equal(tf.argmax(y_model,1),tf.argmax(y_data,1))
            accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
            print(sess.run(accuracy,feed_dict={x_data:mnist.test.images,y_data:mnist.test.labels}))

"""
12-2
提高一点点准确率，
激活函数修正为relu
损失函数修正为交叉熵y=-∑y_data*log(y_model)
"""
def demo2():
    x_data = tf.placeholder(tf.float32, [None, 784])
    weight = tf.Variable(tf.ones([784, 10]))
    bias = tf.Variable(tf.ones([10]))
    y_model = tf.nn.relu(tf.matmul(x_data, weight) + bias)#激活函数修正
    y_data = tf.placeholder(tf.float32, [None, 10])
    loss = -tf.reduce_sum(y_data*tf.log(y_model))
    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(loss)
    init = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init)
    for _ in range(100000):
        batch_xs, batch_ys = mnist.train.next_batch(50)
        sess.run(train_step, feed_dict={x_data: batch_xs, y_data: batch_ys})
        if _ % 50 == 0:
            correct_prediction = tf.equal(tf.argmax(y_model, 1), tf.argmax(y_data, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            print(sess.run(accuracy, feed_dict={x_data: mnist.test.images, y_data: mnist.test.labels}))

"""
12--3
增加深度，
增加一个隐藏层
增加一个影藏层以后准确率下降到0.1，亏了
"""
def dem3():
    x_data = tf.placeholder(tf.float32, [None, 784])
    #第一隐藏层
    weight1 = tf.Variable(tf.ones([784, 256]))
    bias1 = tf.Variable(tf.ones([256]))
    y_model1 = tf.matmul(x_data, weight1) + bias1
    #第二隐藏层
    weight2 = tf.Variable(tf.ones([256, 10]))
    bias2 = tf.Variable(tf.ones([10]))

    y_model = tf.nn.softmax(tf.matmul(y_model1, weight2) + bias2)
    y_data = tf.placeholder(tf.float32, [None, 10])
    loss = -tf.reduce_sum(y_data * tf.log(y_model))
    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(loss)
    init = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init)
    for _ in range(1000):
        batch_xs, batch_ys = mnist.train.next_batch(50)
        sess.run(train_step, feed_dict={x_data: batch_xs, y_data: batch_ys})
        if _ % 50 == 0:
            correct_prediction = tf.equal(tf.argmax(y_model, 1), tf.argmax(y_data, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            print(sess.run(accuracy, feed_dict={x_data: mnist.test.images, y_data: mnist.test.labels}))

"""
    tf.nn.batch_normalization(x, mean, variance, offset, scale, variance_epsilon, name=None) 
    由函数接口可知，tf.nn.moments 计算返回的 mean 和 variance 作为 tf.nn.batch_normalization 参数进一步调用
在这一堆参数里面，其中x，mean和variance这三个，已经知道了，就是通过moments计算得到的，另外菱格参数，offset和scale
一般需要训练，其中offset一般初始化为0，scale初始化为1，另外这两个参数的offset，scale的维度和mean相同
"""
def batch_norm(Wx_plus_b, n_output):
    mean, vars = tf.nn.moments(Wx_plus_b, [0]),#均值，方差
    scale = tf.Variable(tf.random_normal([n_output]))
    beta = tf.Variable(tf.random_normal([n_output]))
    epsilon = 0.001
    Wx_plus_b_out = tf.nn.batch_normalization(Wx_plus_b, mean, vars, beta, scale, epsilon)
    return Wx_plus_b_out
"""
12-4
卷积神经网络
"""
def demo4():
    x_data=tf.placeholder(tf.float32,[None,784])
    x_image=tf.reshape(x_data,[-1,28,28,1])

    #创建一个卷积层
    w_conv=tf.Variable(tf.ones([5,5,1,32]))#卷积核，5*5d的卷积核，输入数据的通道是1，输出数据的通道是32
    b_conv=tf.Variable(tf.ones([32]))
    ##tf.nn.conv2d是TensorFlow里面实现卷积的函数
    # tf.nn.conv2d(input, filter, strides, padding, use_cudnn_on_gpu=None, name=None)
    # 第一个参数input：指需要做卷积的输入图像，它要求是一个Tensor，具有[batch, in_height, in_width, in_channels]这样的shape，具体含义是[训练时一个batch的图片数量, 图片高度, 图片宽度, 图像通道数]，注意这是一个4维的Tensor，要求类型为float32和float64其中之一
    #第二个参数filter：相当于CNN中的卷积核，它要求是一个Tensor，具有[filter_height, filter_width, in_channels, out_channels]这样的shape，具体含义是[卷积核的高度，卷积核的宽度，图像通道数，卷积核个数]，要求类型与参数input相同，有一个地方需要注意，第三维in_channels，就是参数input的第四维
    #第三个参数strides：卷积时在图像每一维的步长，这是一个一维的向量，长度4
    #第四个参数padding：string类型的量，只能是"SAME","VALID"其中之一，这个值决定了不同的卷积方式（后面会介绍）
    #第五个参数：use_cudnn_on_gpu:bool类型，是否使用cudnn加速，默认为true
    #结果返回一个Tensor，这个输出，就是我们常说的feature map，shape仍然是[batch, height, width, channels]这种形式。
    h_conv=tf.nn.relu(tf.nn.conv2d(x_image,w_conv,strides=[1,1,1,1],padding="VALID")+b_conv)

    #池化层
    # 第二个参数ksize：池化窗口的大小，取一个四维向量，一般是[1, height, width, 1]，
    # 因为我们不想在batch和channels上做池化，所以这两个维度设为了1
    #第三个参数strides：和卷积类似，窗口在每一个维度上滑动的步长，一般也是[1, stride, stride, 1]
    h_pool=tf.nn.max_pool(h_conv,ksize=[1,2,2,1],strides=[1,2,2,1],padding="VALID")

    w_fc=tf.Variable(tf.ones([12*12*32,1024]))
    b_fc=tf.Variable(tf.ones([1024]))

    h_pool_falt=tf.reshape(h_pool,[-1,12*12*32])
    h_fc=tf.nn.relu(batch_norm(tf.matmul(h_pool_falt,w_fc)+b_fc,1024))

    w_fc2=tf.Variable(tf.ones([1024,10]))
    b_fc2=tf.Variable(tf.ones([10]))

    y_model=tf.nn.softmax(batch_norm(tf.matmul(h_fc,w_fc2)+b_fc2,10))
    y_data=tf.placeholder(tf.float32,[None,10])

    loss=-tf.reduce_sum(y_data*tf.log(y_model))
    train_step=tf.train.AdamOptimizer(0.01).minimize(loss)
    init=tf.initialize_all_variables()
    sess=tf.Session()
    sess.run(init)

    for _ in range(1000):
        batch_xs,batch_ys=mnist.train.next_batch(200)
        sess.run(train_step,feed_dict={x_data:batch_xs,y_data:batch_ys})
        if _%50 ==0:
            correct_prediction=tf.equal(tf.argmax(y_model,1),tf.argmax(y_data,1))
            accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
            print(sess.run(accuracy, feed_dict={x_data: mnist.test.images, y_data: mnist.test.labels}))




def weight_variable(shape):
    init=tf.truncated_normal(shape=shape,stddev=0.1)
    return tf.Variable(init)

def bias_variable(shape):
    init=tf.constant(0.1,shape=shape)
    return tf.Variable(init)

def conv2d(x,W):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding="VALID")

def max_pool_2x2(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding="VALID")


"""

多层卷积神经网络
"""
def demo5():
    x_data=tf.placeholder(tf.float32,shape=[None,784])
    y_data=tf.placeholder(tf.float32,shape=[None,10])

    W_conv1=weight_variable([5,5,1,32])
    b_conv1=bias_variable([32])
    x_image=tf.reshape(x_data,[-1,28,28,1])
    h_conv1=tf.nn.relu(conv2d(x_image,W_conv1)+b_conv1)
    h_pool1=max_pool_2x2(h_conv1)

    W_conv2=weight_variable([5,5,32,64])
    b_conv2=bias_variable([64])
    h_conv2=tf.nn.relu(conv2d(h_pool1,W_conv2)+b_conv2)
    h_pool2=max_pool_2x2(h_conv2)

    W_fc1=weight_variable([4*4*64,1024])
    b_fc1=bias_variable([1024])

    h_pool2_flat=tf.reshape(h_pool2,[-1,4*4*64])
    h_fc1=tf.nn.relu(tf.matmul(h_pool2_flat,W_fc1)+b_fc1)

    keep_prob=tf.placeholder(tf.float32)
    #tf.nn.dropout(x, keep_prob, noise_shape=None, seed=None, name=None)
    #此函数是为了防止在训练中过拟合的操作，将训练输出按一定规则进行变换
    # tensorflow中的dropout就是：使输入tensor中某些元素变为0，其它没变0的元素变为原来的1/keep_prob大小！

    h_fc1_drop=tf.nn.dropout(h_fc1,keep_prob=keep_prob)

    W_fc2=weight_variable([1024,10])
    b_fc2=bias_variable([10])

    y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop,W_fc2)+b_fc2)
    cross_entropy=-tf.reduce_sum(y_data*tf.log(y_conv))

    train_step=tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    correct_prediction=tf.equal(tf.argmax(y_conv,1),tf.argmax(y_data,1))
    accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

    sess=tf.Session()
    sess.run(tf.initialize_all_variables())

    for i in range(10000):
        batch=mnist.train.next_batch(50)
        if i%5==0:
            train_accuracy=sess.run(accuracy,feed_dict={x_data:batch[0],y_data:batch[1],keep_prob:1.0})
            print("step %d,training accuracy %g"%(i,train_accuracy))
            sess.run(train_step,feed_dict={x_data:batch[0],y_data:batch[1],keep_prob:0.5})


# demo1()
# demo2()
# dem3()
# demo4()
demo5()










