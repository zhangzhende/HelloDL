import tensorflow.examples.tutorials.mnist.input_data as input_data
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import scipy.misc

mnist = input_data.read_data_sets("../MNIST_data/", one_hot=True)

imagepath = "./image/"
"""
对抗生成网络--手写体
思路没问题，但是代码有问题，如果要可用模型，去deep
"""

# 保存图片函数
def save_images(images, size, path=imagepath):
    """
    Save the samples images
    The best size number is
            int(max(sqrt(image.shape[0]),sqrt(image.shape[1]))) + 1
    example:
        The batch_size is 64, then the size is recommended [8, 8]
        The batch_size is 32, then the size is recommended [6, 6]
    """

    # 图片归一化，主要用于生成器输出是 tanh 形式的归一化
    img = (images + 1.0) / 2.0
    h, w = img.shape[1], img.shape[2]

    # 产生一个大画布，用来保存生成的 batch_size 个图像
    merge_img = np.zeros((h * size[0], w * size[1], 3))

    # 循环使得画布特定地方值为某一幅图像的值
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        merge_img[j * h:j * h + h, i * w:i * w + w, :] = image

    # 保存画布
    return scipy.misc.imsave(path, merge_img)


def sampler(z, train=True):
    tf.get_variable_scope().reuse_variables()
    return generate(z, is_training=train)


def discriminate(input_data, scope="discriminate", reuse=False, is_training=True):
    batch_norm_params = {
        "is_training": is_training,
        "decay": 0.9997,
        "epsilon": 0.001,
        "updates_collections": tf.GraphKeys.UPDATE_OPS,
        "variables_collections": {
            "beta": None,
            "gamma": None,
            "moving_mean": ["moving_vars"],
            "moving_variance": ["moving_vars"]
        }
    }
    with tf.variable_scope(scope, "discriminate", [input_data]):
        with slim.arg_scope([slim.conv2d],
                            normalizer_fn=slim.batch_norm,
                            normalizer_params=batch_norm_params,
                            weights_initializer=tf.truncated_normal_initializer(0.0, 0.1),
                            weights_regularizer=slim.l1_l2_regularizer()):
            conv1 = slim.conv2d(inputs=input_data, num_outputs=32, kernel_size=[3, 3], padding="SAME", scope="d_conv1",
                                reuse=reuse)
            conv2 = slim.conv2d(inputs=conv1, num_outputs=64, kernel_size=[5, 5], padding="SAME", scope="d_conv2",
                                reuse=reuse)
            conv3 = slim.conv2d(inputs=conv2, num_outputs=32, kernel_size=[3, 3], padding="SAME", scope="d_conv3",
                                reuse=reuse)
            out = slim.conv2d(inputs=conv3, num_outputs=1, kernel_size=[28, 28], padding="SAME", scope="d_conv4",
                              reuse=reuse)
    return out


def generate(batchSize, trainable=True, scope="generate", reuse=False, is_training=False):
    batch_norm_params = {
        "is_training": is_training,
        "decay": 0.9997,
        "epsilon": 0.001,
        "updates_collections": tf.GraphKeys.UPDATE_OPS,
        "variables_collections": {
            "beta": None,
            "gamma": None,
            "moving_mean": ["moving_vars"],
            "moving_variance": ["moving_vars"]
        }
    }
    with tf.variable_scope(scope, "generate", [batchSize]):
        with slim.arg_scope([slim.conv2d],
                            normalizer_fn=slim.batch_norm,
                            normalizer_params=batch_norm_params,
                            weights_initializer=tf.truncated_normal_initializer(0.0, 0.1),
                            weights_regularizer=slim.l2_regularizer(0.00005)):
            img_x = tf.random_normal([batchSize, 28, 28, 1])
            conv1 = slim.conv2d(inputs=img_x, num_outputs=32, kernel_size=[3, 3], padding="SAME", scope="g_conv1",
                                trainable=trainable, reuse=reuse)
            conv2 = slim.conv2d(inputs=conv1, num_outputs=64, kernel_size=[5, 5], padding="SAME", scope="g_conv2",
                                trainable=trainable, reuse=reuse)
            conv3 = slim.conv2d(inputs=conv2, num_outputs=32, kernel_size=[3, 3], padding="SAME", scope="g_conv3",
                                trainable=trainable, reuse=reuse)
            conv4 = slim.conv2d(inputs=conv3, num_outputs=1, kernel_size=[5, 5], padding="VALID", scope="g_conv4",
                                trainable=trainable, reuse=reuse)
            out = tf.tanh(conv4, name="g_tanh")
    return out


def showMnistGan():
    batch_size = 30
    logPath = "./model2/"
    with tf.Graph().as_default():
        x_input = tf.placeholder(tf.float32, [None, 28, 28, 1])
        # with tf.variable_scope("usr"):
        Gz = generate(batch_size)
        Dx = discriminate(x_input)
        Dg = discriminate(Gz, reuse=True)
        samples = sampler(batch_size)
        # 对生成的图像是真de判定
        d_loss_real = tf.reduce_mean(slim.losses.sigmoid_cross_entropy(multi_class_labels=tf.ones_like(Dx), logits=Dx))
        # 对生成的图像是假的判定
        d_loss_fake = tf.reduce_mean(slim.losses.sigmoid_cross_entropy(multi_class_labels=tf.zeros_like(Dg), logits=Dg))
        d_loss = d_loss_real + d_loss_fake
        # 对生成器的结果进行判定
        g_loss = tf.reduce_mean(slim.losses.sigmoid_cross_entropy(multi_class_labels=tf.ones_like(Dg), logits=Dg))
        tvars = tf.trainable_variables()
        d_vars = [var for var in tvars if "d_" in var.name]
        g_vars = [var for var in tvars if "g_" in var.name]
        d_trainer = tf.train.AdamOptimizer(0.005).minimize(loss=d_loss, var_list=d_vars)
        g_trainer = tf.train.AdamOptimizer(0.001).minimize(loss=g_loss, var_list=g_vars)
        tf.summary.scalar("Generate_loss", g_loss)
        tf.summary.scalar("Discriminator_loss", d_loss)
        image_for_tensorboard = generate(5, is_training=False, reuse=True)
        tf.summary.image("Generate_images", image_for_tensorboard, 5)
        merged_summary_op = tf.summary.merge_all()
        saver = tf.train.Saver()
        with tf.Session() as sess:
            writer = tf.summary.FileWriter(logdir=logPath, graph=sess.graph)
            sess.run(tf.global_variables_initializer())
            for i in range(120):
                batch_xs = mnist.train.next_batch(batch_size)[0].reshape([batch_size, 28, 28, 1])
                _, d_loss_var = sess.run([d_trainer, d_loss], feed_dict={x_input: batch_xs})
                if i % 10 == 1:
                    sample = sess.run(samples, feed_dict={x_input: batch_xs})
                    samples_path = './image/'
                    save_images(sample, [8, 8],
                                samples_path + 'test_%d_epoch.png' % (i))
                    print('save down')
                print("d_loss_var", d_loss_var)
                print("--------pre train epoch %d ebd-----------------" % i)
            i = 0
            while True:
                batch_xs = mnist.train.next_batch(batch_size)[0].reshape([batch_size, 28, 28, 1])
                *total_op, g_loss_var = sess.run([d_trainer, g_trainer, g_loss], feed_dict={x_input: batch_xs})
                print("g_loss_var", g_loss_var)
                i += 1
                if (i + 1) % 3 == 0:
                    merged_summary = sess.run(merged_summary_op, feed_dict={x_input: batch_xs})
                    writer.add_summary(merged_summary, global_step=i)
                    saver.save(sess, "./model2/GAN.ckpt")
                print("-------model train epoch %d end-----------" % i)


showMnistGan()
