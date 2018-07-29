import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import matplotlib.pyplot as plt


"""
Slim的简单啊使用，
MLP多重感知器模型
"""

'''
+++++++++++++++++++++++++++++++++++++++++++
第一部分-模型设计
++++++++++++++++++++++++++++++++++++++++++
'''


def mlp_model(inputs, is_training=True, scope="mlp_model"):
    """
    #创建一个MLP模型
    :param inputs: 一个大小为【batchSize，dimensions】的tensor张亮作为输入数据
    :param is_training: 是否处于训练状态，（当进行使用时模型处于非训练状态，计算时可节省大量时间）
    :param scope: 命名空间的名称
    :return: prediction, end_points，prediction是模型计算的最终值,而end_points用以搜集每层计算值得字典
    """
    with tf.variable_scope(scope, "mlp_model", [inputs]):
        # 使用endpoint记录每一层的输出，，这样做的好处是对每一层都有个记录，方便以后Finetuning
        end_points = {}
        # 创建一个参数空间用以记录使用的各种层和激活函数，以及各种参数的正则化修正
        with slim.arg_scope([slim.fully_connected],
                            activation_fn=tf.nn.relu,
                            weights_regularizer=slim.l2_regularizer(0.01)):
            # 第一个全连接层，输出为32个节点，这里需要注意的是，全连接层所需要的参数的定义，激活函数的使用在前面的arg_scope已经定义过
            net = slim.fully_connected(inputs=inputs, num_outputs=32, scope="fc1")
            end_points["fc1"] = net
            # 使用dropout修正，每次保存0.5
            net = slim.dropout(net, keep_prob=0.5, is_training=is_training)
            # 第二个全连接层，输出为16个节点
            net = slim.fully_connected(net, num_outputs=16, scope="fc2")
            end_points["fc2"] = net
            # 使用dropout修正，每次保存0.5
            net = slim.dropout(net, keep_prob=0.5, is_training=is_training)
            # 使用一个全连接层作为最终的计算，输出一个值，不适用激活函数
            prediction = slim.fully_connected(net, num_outputs=1, activation_fn=None, scope="prediction")
            end_points["out"] = prediction
            return prediction, end_points


'''
+++++++++++++++++++++++++++++++++++++++++++
第二部分-验证模型结构
++++++++++++++++++++++++++++++++++++++++++
'''


def validate():
    with tf.Graph().as_default():
        # 定义两个占位符用作输入输出
        inputs = tf.placeholder(dtype=tf.float32, shape=(None, 1))
        outputs = tf.placeholder(dtype=tf.float32, shape=(None, 1))

        # 使用MLP模型计算输入值
        prediction, end_point = mlp_model(inputs=inputs)

        # 打印模型计算值
        print(prediction)

        # 打印每层的名称以及计算值
        print("layers")
        for name, value in end_point.items():
            print("name={},Node={}".format(name, value))

        print("\n")
        print("---------------------")
        print("Parameters")
        for v in slim.get_model_variables():
            print("name ={},shape={}".format(v.name, v.get_shape()))


'''
+++++++++++++++++++++++++++++++++++++++++++
第三部分-创建数据集
++++++++++++++++++++++++++++++++++++++++++
'''


def produce_batch(batch_size, noise=0.3):
    xs = np.random.random(size=[batch_size, 1]) * 10
    ys = np.cos(xs) + 5 + np.random.normal(size=[batch_size, 1], scale=noise)
    return [xs.astype(np.float32), ys.astype(np.float32)]


def show():
    x_train, y_train = produce_batch(200)
    x_test, y_test = produce_batch(200)
    plt.scatter(x_train, y_train, marker="8")
    plt.scatter(x_test, y_test, marker="*")
    plt.show()


"""
数据转化为tensor格式
tf.constant转换数据格式
注意转化步骤需要在一个会话中完成
"""


def convert_data_to_tensors(x, y):
    inputs = tf.constant(x)
    inputs.set_shape([None, 1])
    outputs = tf.constant(y)
    outputs.set_shape([None, 1])
    return inputs, outputs


'''
+++++++++++++++++++++++++++++++++++++++++++
第四部-模型的训练
++++++++++++++++++++++++++++++++++++++++++
'''


def train(prediction, outputs):
    loss = slim.losses.mean_squared_error(prediction, outputs, scope="loss")  # 均方差
    # 使用梯度下降算法训练模型
    optimizer = slim.train.GradientDescentOptimizer(0.005)
    train_op = slim.learning.create_train_op(loss, optimizer)


import shutil

save_path = "./model"


def demoTrain():
    """
    训练模型
    :return: 
    """
    shutil.rmtree(save_path)
    g = tf.Graph()
    with g.as_default():
        # 在控制台打印log日志信息
        tf.logging.set_verbosity(tf.logging.INFO)
        # 创建数据集
        xs, ys = produce_batch(200)
        # 将数据转化为tensor
        inputs, outputs = convert_data_to_tensors(xs, ys)
        # 计算模型值
        prediction, _ = mlp_model(inputs)
        # 损失函数定义
        loss = slim.losses.mean_squared_error(prediction, outputs, scope="loss")
        # 使用梯度下降算法训练模型
        optimizer = slim.train.GradientDescentOptimizer(0.005)
        train_op = slim.learning.create_train_op(loss, optimizer)
        # 使用tensorflow高级执行框架“图”去执行模型训练任务
        final_loss = slim.learning.train(train_op=train_op,
                                         logdir=save_path,
                                         number_of_steps=1000,
                                         log_every_n_steps=200)
        print("Finished training .last batch loss:", final_loss)
        print("checkpoint saved in %s" % save_path)

def demoTrainWithMultiLoss():
    """
    训练模型,使用“图”去执行训练
    :return: 
    """
    shutil.rmtree(save_path)
    g = tf.Graph()
    with g.as_default():
        # 在控制台打印log日志信息
        tf.logging.set_verbosity(tf.logging.INFO)
        # 创建数据集
        xs, ys = produce_batch(200)
        # 将数据转化为tensor
        inputs, outputs = convert_data_to_tensors(xs, ys)
        # 计算模型值
        prediction, _ = mlp_model(inputs)
        # 损失函数定义-均方差
        loss = slim.losses.mean_squared_error(prediction, outputs, scope="loss")
        absolute_loss=slim.losses.absolute_difference(prediction,outputs,scope="absolute_loss")
        total_loss=loss+absolute_loss
        # 使用梯度下降算法训练模型
        optimizer = slim.train.GradientDescentOptimizer(0.005)
        train_op = slim.learning.create_train_op(total_loss, optimizer)
        # 使用tensorflow高级执行框架“图”去执行模型训练任务
        final_loss = slim.learning.train(train_op=train_op,
                                         logdir=save_path,
                                         number_of_steps=1000,
                                         log_every_n_steps=200)
        print("Finished training .last batch loss:", final_loss)
        print("checkpoint saved in %s" % save_path)

def demoTrainWithMultiLossBySession():
    """
    训练模型,使用传统的tf.Session()去执行训练
    :return: 
    """
    shutil.rmtree(save_path)
    g = tf.Graph()
    with g.as_default():
        # 在控制台打印log日志信息
        tf.logging.set_verbosity(tf.logging.INFO)
        # 创建数据集
        xs, ys = produce_batch(200)
        # 将数据转化为tensor
        inputs, outputs = convert_data_to_tensors(xs, ys)
        # 计算模型值
        prediction, _ = mlp_model(inputs)
        # 损失函数定义-均方差
        loss = slim.losses.mean_squared_error(prediction, outputs, scope="loss")
        absolute_loss=slim.losses.absolute_difference(prediction,outputs,scope="absolute_loss")
        total_loss=loss+absolute_loss
        # 使用梯度下降算法训练模型
        optimizer = slim.train.GradientDescentOptimizer(0.005)
        train_op = slim.learning.create_train_op(total_loss, optimizer)
        # 使用tensorflow高级执行框架“图”去执行模型训练任务
        saver =slim.train.Saver()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for _ in range(1000):
                sess.run(train_op)
            saver.save(sess,save_path+"MLP_train_multiple_loss.ckpt")
            print(sess.run(total_loss))

'''
+++++++++++++++++++++++++++++++++++++++++++
复用模型
++++++++++++++++++++++++++++++++++++++++++
'''
def reuseModel(save_path):
    with tf.Graph().as_default():
        # 在控制台打印log日志信息
        tf.logging.set_verbosity(tf.logging.INFO)
        #创建数据集
        xs,ys=produce_batch(200)
        #将数据转化为tensor
        inputs,outputs=convert_data_to_tensors(xs,ys)
        #计算模型值
        prediction,_=mlp_model(inputs,is_training=False)
        saver=tf.train.Saver()
        save_path=tf.train.latest_checkpoint(save_path)
        with tf.Session() as sess:
            saver.restore(sess,save_path)
            inputs,prediction,outputs=sess.run([inputs,prediction,outputs])

'''
+++++++++++++++++++++++++++++++++++++++++++
MLP模型的评估
19-9
++++++++++++++++++++++++++++++++++++++++++
'''
def preform():
    with tf.Graph().as_default():
        # 在控制台打印log日志信息
        tf.logging.set_verbosity(tf.logging.INFO)
        #创建数据集
        xs,ys=produce_batch(200)
        #将数据转化为tensor
        inputs,outputs=convert_data_to_tensors(xs,ys)
        #计算模型值
        prediction,_=mlp_model(inputs,is_training=False)
        #指定的度量值-相对误差和绝对误差
        name_to_value_nodes,name_to_update_nodes=slim.metrics.aggregate_metric_map({
            "Mean Squared Error":slim.metrics.streaming_mean_squared_error(prediction,outputs),
            "Mean absolute Error":slim.metrics.streaming_mean_absolute_error(prediction,outputs)
        })
        sv=tf.train.Supervisor(logdir=save_path)
        with sv.managed_session() as sess:
            name_to_value=sess.run(name_to_value_nodes)
            name_to_update=sess.run(name_to_update_nodes)

        for key,value in name_to_value.items():
            print((key,value))
        print("\n")
        for key ,value in name_to_update.items():
            print((key,value))

'''
+++++++++++++++++++++++++++++++++++++++++++
调用
++++++++++++++++++++++++++++++++++++++++++
'''
# validate()
# show()
# demoTrain()
# demoTrainWithMultiLoss()
# demoTrainWithMultiLossBySession()
preform()