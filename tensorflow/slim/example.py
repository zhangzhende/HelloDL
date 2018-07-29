import tensorflow as tf
import tensorflow.contrib.slim as slim
"""
Slim 的简单使用

"""

"""
slim 定义变量
tf.truncated_normal_initializer(stddev=0.1)：截断正态分布随机数，均值mean默认0，标准差stddev0.1
slim.l2_regularizer：使用L2范数进行正则化处理
"""
weights = slim.variable(name="weights", shape=[1, 217, 217, 3], dtype=tf.float32,
                        initializer=tf.truncated_normal_initializer(stddev=0.1),
                        regularizer=slim.l2_regularizer(scale=0.05), device="/CPU:0")
"""
模型变量
19-1
"""
def modelVariable():
    weight1=slim.model_variable(name="weight1",
                                shape=[2,3],
                                initializer=tf.truncated_normal_initializer(stddev=0.1),
                                regularizer=slim.l2_regularizer(scale=0.05))
    weight2=slim.model_variable(name="weight2",
                                shape=[2,3],
                                initializer=tf.truncated_normal_initializer(stddev=0.1),
                                regularizer=slim.l2_regularizer(scale=0.05))
    model_variable=slim.get_model_variables()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        print(sess.run(weight1))
        print("-----------------")
        print(sess.run(weight2))
        print("----------------")
        print(sess.run(slim.get_variables_by_suffix("weight1")))

"""
19-2
普通变量
"""
def normolVariable():
    weight1=slim.variable(name="weight1",
                          shape=[2,3],
                          initializer=tf.truncated_normal_initializer(stddev=0.1),
                          regularizer=slim.l2_regularizer(scale=0.05))
    weight2 = slim.variable(name="weight2",
                            shape=[2, 3],
                            initializer=tf.truncated_normal_initializer(stddev=0.1),
                            regularizer=slim.l2_regularizer(scale=0.05))
    variable=slim.get_variables()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        print(sess.run(weight1))
        print("-----------------")
        print(sess.run(weight2))
        print("----------------")
        print(sess.run(variable))

"""
19-3
自定义变量交由slim进行统一管理使用
例tf创建通用变量weight，然后加入slim
"""
def variableUse():
    weight=tf.Variable(tf.ones([2,3]))
    slim.add_model_variable(weight)
    modelVariable=slim.get_model_variables()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print(sess.run(modelVariable))

# modelVariable()
normolVariable()