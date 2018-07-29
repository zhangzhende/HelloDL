import os
import cv2
import tensorflow as tf
import numpy as np
from PIL import Image

"""简单使用队列 
with-as的使用可以使他自动关闭
"""


def useQueue():
    with tf.Session() as sess:
        q = tf.FIFOQueue(3, "float")  # 定义队列，3个位置，float类型
        init = q.enqueue_many(([0.1, 0.2, 0.3],))  # 填充三个数据
        init2 = q.dequeue()  # 弹出一个数据
        init3 = q.enqueue(1.)  # 压入一个数据

        sess.run(init)  # 执行操作
        sess.run(init2)
        sess.run(init3)
        """所有的操作都在run里面执行"""
        quelen = sess.run(q.size())
        for i in range(quelen):
            print(sess.run(q.dequeue()))


"""多线程，异步加载，一方面启动两个线程同时往队列中添加数据"""


def duoxianchengQueue():
    with tf.Session() as sess:
        q = tf.FIFOQueue(1000, "float32")  # 创建队列
        counter = tf.Variable(0.0)  # 创建变量
        add_op = tf.assign_add(counter, tf.constant(1.0))  # 定义操作，变量值加1
        enqueueData_op = q.enqueue(counter)  # 定义操作，变量值添加到队列中
        # 线程面向队列q,启动两个线程，每个线程中【add_op，enqueueData_op】两个操作
        qr = tf.train.QueueRunner(queue=q, enqueue_ops=[add_op, enqueueData_op] * 2)
        sess.run(tf.initialize_all_variables())
        enqueue_threads = qr.create_threads(sess, start=True)  # 启动入队线程qr

        for i in range(10):
            # 弹出队列，因为是异步的所以出来的不一定是连续的数，同时有可能输出完了准备main，但是队列那块没有关，这时候，队列那边会报错【ERROR:tensorflow:Exception in QueueRunner: Session has been closed.】
            print(sess.run(q.dequeue()))


"""多线程，异步加载，但是程序不报错，会一直挂起"""


def duoxianchengQueue2():
    q = tf.FIFOQueue(1000, "float32")  # 创建队列
    counter = tf.Variable(0.0)  # 创建变量
    add_op = tf.assign_add(counter, tf.constant(1.0))  # 定义操作，变量值加1
    enqueueData_op = q.enqueue(counter)  # 定义操作，变量值添加到队列中
    sess = tf.Session()
    # 线程面向队列q,启动两个线程，每个线程中【add_op，enqueueData_op】两个操作
    qr = tf.train.QueueRunner(queue=q, enqueue_ops=[add_op, enqueueData_op] * 2)
    sess.run(tf.initialize_all_variables())
    enqueue_threads = qr.create_threads(sess, start=True)  # 启动入队线程qr

    for i in range(10):
        # 弹出队列，因为是异步的所以出来的不一定是连续的数，没有报错，但是程序会一直挂起
        print(sess.run(q.dequeue()))


def xiangchengStartStop():
    q = tf.FIFOQueue(1000, "float32")  # 创建队列
    counter = tf.Variable(0.0)  # 创建变量
    add_op = tf.assign_add(counter, tf.constant(1.0))  # 定义操作，变量值加1
    enqueueData_op = q.enqueue(counter)  # 定义操作，变量值添加到队列中
    sess = tf.Session()
    # 线程面向队列q,启动两个线程，每个线程中【add_op，enqueueData_op】两个操作
    qr = tf.train.QueueRunner(queue=q, enqueue_ops=[add_op, enqueueData_op] * 2)
    sess.run(tf.initialize_all_variables())
    # enqueue_threads = qr.create_threads(sess, start=True)  # 启动入队线程qr
    # 线程协调器
    coord = tf.train.Coordinator()
    enqueue_threads = qr.create_threads(sess=sess, coord=coord, start=True)

    for i in range(10):
        # 弹出队列，因为是异步的所以出来的不一定是连续的数，没有报错，但是程序会一直挂起
        print(sess.run(q.dequeue()))
    # 请求线程停止。
    coord.request_stop()
    # 等待被指定的线程终止
    coord.join(enqueue_threads)


"""
创建CSV文件
"""


def CVSwenjianchuangjian():
    path = 'jpg'
    filenames = os.listdir(path=path)
    strText = ""

    with open("train_list.csv", "w") as fid:
        for a in range(len(filenames)):
            pathb = path + os.sep + filenames[a]
            filenamesb = os.listdir(path=pathb)
            for b in range(len(filenamesb)):
                strText = pathb + os.sep + filenamesb[b] + "," + "1" + "\n"
                print(strText)
                fid.write(strText)
    fid.close()


"""
读取CSV文件
"""


def duquCSVwenjian():
    image_add_list = []
    image_label_list = []
    with open("train_list.csv", "r") as fid:
        for image in fid.readlines():
            image_add_list.append(image.strip().split(",")[0])
            image_label_list.append(image.strip().split(",")[1])
    print(image_add_list)
    print(image_label_list)
    # 展示图片
    image = tf.image.convert_image_dtype(tf.image.decode_jpeg(tf.read_file("jpg/001/68.jpg"), channels=1),
                                         dtype=tf.float32)
    with tf.Session() as sess:
        cv2Img = sess.run(image)
        # 将图片压缩转换到指定大小，像素*像素
        image2 = cv2.resize(cv2Img, (400, 400))
        cv2.imshow("image", image2)
        cv2.waitKey()


"""
按照文件路径读取图片
"""


def get_image(image_path):
    img = tf.image.convert_image_dtype(tf.image.decode_jpeg(tf.read_file(image_path), channels=1), dtype=tf.uint8)
    return img


"""
TFRecord文件创建，TFRecord文件格式是TensorFlow专用的，
"""


def TFrecordswenjianchuangjian():
    # for _ in range(10):[[0.65961324 0.6992843  0.48007064]
    # [0.53651416 0.29772888 0.13130141]]
    # 创建一个两行三列的矩阵
    #     randomArray = np.random.random((2, 3))
    #     print(randomArray)
    writer = tf.python_io.TFRecordWriter("trainArray.tfrecords")
    for _ in range(100):
        randomArray = np.random.random((1, 3))  # 创建一个1行3列的矩阵
        # array_raw=randomArray.tobytes()
        array_raw = randomArray.tostring()
        # Feature有三种格式list，【int64_list，bytes_list，float_list，写死的】
        example = tf.train.Example(features=tf.train.Features(feature={
            "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[0])),
            "img_raw": tf.train.Feature(bytes_list=tf.train.BytesList(value=[array_raw]))
        }))
        writer.write(example.SerializeToString())
    writer.close()


"""
给TFrecordswenjianduqu函数造数据
"""


def TFRecordwenjian():
    a_data = 0.834
    b_data = [17]
    c_data = np.array([[0, 1, 2], [3, 4, 5]])
    c = c_data.astype(np.uint8)
    c_raw = c.tostring()
    example = tf.train.Example(features=tf.train.Features(
        feature={'a': tf.train.Feature(float_list=tf.train.FloatList(value=[a_data])),
                  'b': tf.train.Feature(int64_list=tf.train.Int64List(value=b_data)),
                  'c': tf.train.Feature(bytes_list=tf.train.BytesList(value=[c_raw]))
                  }  ))
    writer = tf.python_io.TFRecordWriter("trainTest.tfrecords")
    writer.write(example.SerializeToString())
    writer.close()


"""
TFRecord文件读取
"""


def TFrecordswenjianduqu():
    filename_queue = tf.train.string_input_producer(["trainTest.tfrecords"], num_epochs=None)
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example, features={
        'a':tf.FixedLenFeature([],tf.float32),
        'b':tf.FixedLenFeature([],tf.int64),
        'c':tf.FixedLenFeature([],tf.string)
    })
    a=features["a"]
    b=features["b"]
    c_raw=features["c"]
    c=tf.decode_raw(c_raw,tf.uint8)
    c=tf.reshape(c,[2,3])

    a_batch,b_batch,c_batch=tf.train.shuffle_batch([a,b,c],batch_size=1,capacity=200,min_after_dequeue=100,num_threads=2)
    sess = tf.Session()
    init = tf.initialize_all_variables()
    sess.run(init)
    tf.train.start_queue_runners(sess=sess)
    a_val, b_val,c_val = sess.run([a_batch, b_batch,c_batch])

    print(a_val)
    print("+" * 30)
    print(b_val)
    print("+" * 30)
    print(c_val)

"""
10-10
将路径下的图片转化为tfrecords保存在文件中
"""
def parseImageToTfrecords():
    path="jpg"
    filenames=os.listdir(path)
    writer=tf.python_io.TFRecordWriter("train.tfrecords")

    for name in os.listdir(path):
        class_path=path +os.sep+name
        for img_name in os.listdir(class_path):
            img_path=class_path+os.sep+img_name
            img=Image.open(img_path)
            img=img.resize((500,500))
            img_raw=img.tobytes()
            example=tf.train.Example(
                features=tf.train.Features(
                    feature={
                        "label":tf.train.Feature(int64_list=tf.train.Int64List(value=[int(name)])),
                        "image":tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
                    }
                )
            )
            writer.write(example.SerializeToString())
"""
10-12
tfrecords转图片并展示
"""
def parseTfrecordsToImg():
    filename="train.tfrecords"
    filename_queue=tf.train.string_input_producer([filename])
    reader=tf.TFRecordReader()
    _,serialized_example=reader.read(filename_queue)
    features=tf.parse_single_example(
        serialized_example,
        features={
            "label":tf.FixedLenFeature([],tf.int64),
            "image":tf.FixedLenFeature([],tf.string)
        }
    )
    for serialized_examples in tf.python_io.tf_record_iterator("train.tfrecords"):
        example=tf.train.Example()
        example.ParseFromString(serialized_examples)
        image=example.features.feature["image"].bytes_list.value
        label=example.features.feature["label"].int64_list.value
        print(label)
    img=tf.decode_raw(features["image"],tf.uint8)
    img=tf.reshape(img,[500,500,3])
    sess=tf.Session()
    init=tf.initialize_all_variables()
    sess.run(init)
    threads=tf.train.start_queue_runners(sess=sess)
    img=tf.cast(img,tf.float32)*(1./128)-0.5
    label=tf.cast(features["label"],tf.float32)
    imgcv2=sess.run(img)
    cv2.imshow("cool",imgcv2)
    cv2.waitKey()

"""
读取records文件并转换成标签，图片
"""
def readAndDecode(filename):
    filename_queue = tf.train.string_input_producer([filename])
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        features={
            "label": tf.FixedLenFeature([], tf.int64),
            "image": tf.FixedLenFeature([], tf.string)
        }
    )
    img = tf.decode_raw(features["image"], tf.uint8)
    img = tf.reshape(img, [500, 500, 3])
    img = tf.cast(img, tf.float32) * (1. / 128) - 0.5
    label = tf.cast(features["label"], tf.float32)
    return img,label

"""
循环输出10个
"""
def useShuffleBatch():
    filename="train.tfrecords"
    img,label=readAndDecode(filename)
    imgBatch,labelBatch=tf.train.shuffle_batch(
        [img,label],batch_size=1,capacity=10,min_after_dequeue=1
    )
    init=tf.initialize_all_variables()
    sess=tf.Session()
    sess.run(init)
    threads=tf.train.start_queue_runners(sess=sess)
    for _ in range(10):
        val=sess.run(imgBatch)
        label=sess.run(labelBatch)
        val.resize((500,500,3))
        cv2.imshow("cool",val)
        cv2.waitKey()
        print(label)



# useQueue()
# duoxianchengQueue()
# duoxianchengQueue2()
# xiangchengStartStop()
# CVSwenjianchuangjian()
# duquCSVwenjian()
# TFrecordswenjianchuangjian()
# TFRecordwenjian()
# TFrecordswenjianduqu()
# parseImageToTfrecords()
# parseTfrecordsToImg()
useShuffleBatch()