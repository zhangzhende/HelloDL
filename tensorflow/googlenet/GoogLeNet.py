import tensorflow as tf


"""
17-2
googleNet模型
"""
def inception_unit(input_data,weights,biases):
    inception_in=input_data
    #conv 1x1+S1
    inception_1x1_S1=tf.nn.conv2d(input=inception_in,filter=weights["inception_1x1_S1"],strides=[1,1,1,1],padding="SAME")
    inception_1x1_S1=tf.nn.bias_add(inception_1x1_S1,bias=biases["inception_1x1_S1"])
    inception_1x1_S1=tf.nn.relu(inception_1x1_S1)

    # conv 3x3+S1
    inception_3x3_S1_reduce=tf.nn.conv2d(inception_in,weights["inception_3x3_S1_reduce"],strides=[1,1,1,1],padding="SAME")
    inception_3x3_S1_reduce=tf.nn.bias_add(inception_3x3_S1_reduce,bias=biases["inception_3x3_S1_reduce"])
    inception_3x3_S1_reduce=tf.nn.relu(inception_3x3_S1_reduce)
    inception_3x3_S1=tf.nn.conv2d(inception_3x3_S1_reduce,weights["inception_3x3_S1"],strides=[1,1,1,1],padding="SAME")
    inception_3x3_S1=tf.nn.bias_add(inception_3x3_S1,bias=biases["inception_3x3_S1"])
    inception_3x3_S1=tf.nn.relu(inception_3x3_S1)

    # conv 5x5+S1
    inception_5x5_S1_reduce = tf.nn.conv2d(inception_in, weights["inception_5x5_S1_reduce"], strides=[1, 1, 1, 1],
                                           padding="SAME")
    inception_5x5_S1_reduce = tf.nn.bias_add(inception_5x5_S1_reduce, bias=biases["inception_5x5_S1_reduce"])
    inception_5x5_S1_reduce = tf.nn.relu(inception_5x5_S1_reduce)
    inception_5x5_S1 = tf.nn.conv2d(inception_5x5_S1_reduce, weights["inception_5x5_S1"], strides=[1, 1, 1, 1],
                                    padding="SAME")
    inception_5x5_S1 = tf.nn.bias_add(inception_5x5_S1, bias=biases["inception_5x5_S1"])
    inception_5x5_S1 = tf.nn.relu(inception_5x5_S1)

    # maxpool
    inception_MaxPool=tf.nn.max_pool(inception_in,ksize=[1,3,3,1],strides=[1,1,1,1],padding="SAME")
    inception_MaxPool=tf.nn.conv2d(inception_MaxPool,weights["inception_MaxPool"],strides=[1,1,1,1],padding="SAME")
    inception_MaxPool=tf.nn.bias_add(inception_MaxPool,bias=biases["inception_MaxPool"])
    inception_MaxPool=tf.nn.relu(inception_MaxPool)

    inception_out=tf.concat(concat_dim=3,values=[inception_1x1_S1,inception_3x3_S1,inception_5x5_S1,inception_MaxPool])

    return inception_out