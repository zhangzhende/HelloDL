import os
import tensorflow as tf
from PIL import Image

"""
slim的简单使用
"""

"""
19-10
"""
def createRecord(path="../CatVsDog/catanddog",classes={"cat","dog"},imgHeight=224,imgWidth=224):
    """
    生成Tfrecords格式文件
    :param path: 主文件夹位置
    :param classes: 子文件夹名称，每个文件夹的名称作为一个分类
    :param imgHeight: 图片高度
    :param imgWidth: 图片宽度
    :return: 
    """
    writer=tf.python_io.TFRecordWriter("./records/train.tfrecords")
    for index ,name in enumerate(classes):
        classPath=path+"/"+name+"/"
        for imgName in os.listdir(classPath):#返回指定的文件夹包含的文件或文件夹的名字的列表。这个列表以字母顺序。 它不包括 '.' 和'..' 即使它在文件夹中。
            imgPath=classPath+imgName
            img=Image.open(imgPath)
            img=img.resize((imgHeight,imgWidth))
            imgRaw=img.tobytes()
            example=tf.train.Example(
                features=tf.train.Features(
                    features={
                        "label":tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),
                        "imgRaw":tf.train.Feature(bytes_list=tf.train.BytesList(value=[imgRaw]))
                    }
                )
            )
            writer.write(example.SerializeToString())
    writer.close()


"""
19-11
"""
def readAndDecode(filename,imgHeight=224,imgWidth=224):
    """
    独立读取上面生成的tfrecords文件，一次一个
    :param filename: 
    :param imgHeight: 
    :param imgWidth: 
    :return: img ,label
    """
    #创建文件队列，不限读取的数量
    filenameQueue=tf.train.string_input_producer([filename])
    reader =tf.TFRecordReader()
    #reader从文件队列中读入一个序列化样本
    serializedExample=reader.read(filenameQueue)
    #解析符号化的样本
    features=tf.parse_single_example(
        serialized=serializedExample,
        features={
            "label":tf.FixedLenFeature(shape=[],dtype=tf.int64),
            "imgRaw":tf.FixedLenFeature(shape=[],dtype=tf.string)
        }
    )
    label=features["label"]
    img=features["imgRaw"]
    img=tf.decode_raw(bytes=img,out_type=tf.uint8)
    img=tf.reshape(tensor=img,shape=[imgHeight,imgWidth,3])
    img=tf.cast(x=img,dtype=tf.float32)*(1./255)-0.5
    label=tf.cast(label,tf.int32)
    return img,label

def batchReadAndDecode(filename,imgHeight=224,imgWidth=224,batchSize=100):
    """
    批量读取tfrecords文件里的数据
    :param filename: 
    :param imgHeight: 
    :param imgWidth: 
    :param batchSize: 
    :return: 
    """
    #创建文件队列
    filenameQueue=tf.train.string_input_producer([filename],shuffle=True)
    reader=tf.TFRecordReader()
    serializedExample = reader.read(filenameQueue)
    features = tf.parse_single_example(
        serialized=serializedExample,
        features={
            "label": tf.FixedLenFeature(shape=[], dtype=tf.int64),
            "imgRaw": tf.FixedLenFeature(shape=[], dtype=tf.string)
        }
    )
    label = features["label"]
    img = features["imgRaw"]
    img = tf.decode_raw(bytes=img, out_type=tf.uint8)
    img = tf.reshape(tensor=img, shape=[imgHeight, imgWidth, 3])
    img = tf.cast(x=img, dtype=tf.float32) * (1. / 255) - 0.5
    minAfterDequeue=batchSize*9
    capacity=minAfterDequeue+batchSize
    #预取图像和label并且随机打乱，组成batch，此时，tensorrank发生了变化，多了一个batch大小的维度
    exampleBatch,laabelBatch=tf.train.shuffle_batch(tensors=[img,label],batch_size=batchSize,capacity=capacity,min_after_dequeue=minAfterDequeue)
    return exampleBatch,laabelBatch

def showBatchReadAndDecode():
    """
    测试调用批量读取
    :return: 
    """
    init =tf.initialize_all_variables()
    exampleBatch,labelBatch=batchReadAndDecode("./records/train.tfrecords")
    with tf.Session() as sess:
        sess.run(init)
        coord=tf.train.Coordinator()
        threads=tf.train.start_queue_runners(coord=coord)

        for i in range(100):
            example,label=sess.run([exampleBatch,labelBatch])
            print(example[0][112],label)
            print("------------%i--------------"%i)

    coord.request_stop()
    coord.join()































