import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import matplotlib.pyplot as plt

size=500
length=1000
logPath="./model/"

"""
对抗生成网络--模拟正太分布--思路应该没问题，但是代码有问题
"""



def sampleData(size=size,length=length):
    data=[]
    for i in range(size):
        data.append(sorted(np.random.normal(4,1.5,length)))
    return np.array(data).astype(np.float32)

def randomData(size=size,length=length):
    data=[]
    for i in range(size):
        data.append(np.random.random(length))
    return np.array(data).astype(np.float32)

def generate(inputData,reuse=False):
    """
    生成器    
    :param inputData: 
    :param reuse: 
    :return: 
    """
    with tf.variable_scope("generate"):
        """
        arg_scope操作，这一操作符可以让定义在这一scope中的操作共享参数，即如不制定参数的话，则使用默认参数。且参数可以被局部覆盖。使得代码更加简洁
        """
        with slim.arg_scope([slim.fully_connected],
                            weights_initializer=tf.truncated_normal_initializer(0.0,0.1),
                            weights_regularizer=slim.l1_l2_regularizer(),
                            activation_fn=None):
            fc1=slim.fully_connected(inputs=inputData,num_outputs=length,scope="g_fc1",reuse=reuse)
            fc1=tf.nn.softplus(features=fc1,name="g_softplus")
            fc2=slim.fully_connected(inputs=fc1,num_outputs=length,scope="g_fc2",reuse=reuse)
    return fc2

def discriminate(inputData,reuse=False):
    """
    判别器
    :param inputData: 
    :param reuse: 
    :return: 
    """
    with tf.variable_scope("discriminate"):
        with slim.arg_scope([slim.fully_connected],
                            weights_initializer=tf.truncated_normal_initializer(0.0, 0.1),
                            weights_regularizer=slim.l1_l2_regularizer(),
                            activation_fn=None
                            ):
            fc1 = slim.fully_connected(inputs=inputData, num_outputs=length, scope="d_fc1", reuse=reuse)
            fc1=tf.tanh(fc1)
            fc2=slim.fully_connected(inputs=fc1,num_outputs=length,scope="d_fc2",reuse=reuse)
            fc2=tf.tanh(fc2)
            fc3=slim.fully_connected(inputs=fc2,num_outputs=1,scope="d_fc3",reuse=reuse)
            fc3=tf.tanh(fc3)
            fc3=tf.sigmoid(fc3)
    return fc3

def gan():
    with tf.Graph().as_default():
        fakeInput=tf.placeholder(dtype=tf.float32,shape=[size,length],name="fakeInput")
        realInput=tf.placeholder(dtype=tf.float32,shape=[size,length],name="realInput")
        Gz=generate(fakeInput)
        Dz_r=discriminate(realInput)
        Dz_f=discriminate(Gz,reuse=False)

        d_loss=tf.reduce_mean(-tf.log(Dz_r)-tf.log(1-Dz_f))
        g_loss=tf.reduce_mean(-tf.log(Dz_f))

        tf.summary.scalar("Generator_loss",g_loss)
        tf.summary.scalar("Discriminator_loss",d_loss)

        tvars=tf.trainable_variables()
        d_vars=[var for var in tvars if "d_" in var.name]
        g_vars=[var for var in tvars if "g_" in var.name]

        d_optimizator=tf.train.AdamOptimizer(0.0005).minimize(loss=d_loss,var_list=d_vars)
        g_optimizator=tf.train.AdamOptimizer(0.0003).minimize(loss=g_loss,var_list=g_vars)

        merged_summary_op=tf.summary.merge_all()
        saver =tf.train.Saver()
        with tf.Session() as sess:
            writer=tf.summary.FileWriter(logdir=logPath,graph=sess.graph)
            sess.run(tf.global_variables_initializer())

            for i in range(10):
                sess.run(d_optimizator,feed_dict={realInput:sampleData(),fakeInput:randomData()})
                print("--------------------pre_train %d epoch end -----------"%i)

                if i%50==0:
                    merged_summary=sess.run(merged_summary_op,feed_dict={realInput:sampleData(),fakeInput:randomData()})
                    writer.add_summary(merged_summary,global_step=i)
                    saver.save(sess,save_path=logPath,global_step=i)
            for i in range(10):
                sess.run([d_optimizator],feed_dict={realInput:sampleData(),fakeInput:randomData()})
                sess.run([g_optimizator],feed_dict={fakeInput:randomData()})
                print("--------model_train %d epoch end--------------------"%i)

                if i%50 ==0:
                    merged_summary=sess.run(merged_summary_op,feed_dict={realInput:sampleData(),fakeInput:randomData()})
                    writer.add_summary(merged_summary,global_step=i)
                    saver.save(sess,save_path=logPath,global_step=i)

            createData=sess.run(Dz_f,feed_dict={fakeInput:randomData()})
            # realdata=sess.run(Dz_r,feed_dict={realInput:sampleData()})
            # print(createData)
            plt.plot(sampleData(),"b")
            plt.plot(createData,"r")
            plt.show()
gan()






















