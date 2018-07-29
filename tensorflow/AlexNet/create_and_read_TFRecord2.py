import os
import numpy as np
import tensorflow as tf
import cv2
import io

"""获取文件，返回图片列表和标签列表"""


def get_file(filedir):
    images = []
    temp = []
    # TODO walk，这里实际上是遍历了全文件夹
    for root, sub_folders, files in os.walk(filedir):
        """
        os.walk(top[, topdown=True[, onerror=None[, followlinks=False]]])
        top -- 是你所要遍历的目录的地址, 返回的是一个三元组(root,dirs,files)。
        root 所指的是当前正在遍历的这个文件夹的本身的地址
        dirs 是一个 list ，内容是该文件夹中所有的目录的名字(不包括子目录)
        files 同样是 list , 内容是该文件夹中所有的文件(不包括子目录)
        topdown --可选，为 True，则优先遍历 top 目录，否则优先遍历 top 的子目录(默认为开启)。如果 topdown 参数为 True，walk 会遍历top文件夹，与top 文件夹中每一个子目录。
        onerror -- 可选， 需要一个 callable 对象，当 walk 需要异常时，会调用。
        followlinks -- 可选， 如果为 True，则会遍历目录下的快捷方式(linux 下是 symbolic link)实际所指的目录(默认关闭)。
        """
        for name in files:
            #  jion,将多个路径组合后返回
            # os.path.join('/hello/','good/boy/','doiido')
            # '/hello/good/boy/doiido'
            images.append(os.path.join(root, name))  # 图片的路径
        for name in sub_folders:
            temp.append(os.path.join(root, name))
        print(files)

    labels = []
    for one_folder in temp:
        n_img = len(os.listdir(one_folder))  # 返回指定路径下的文件和文件夹列表
        letter = one_folder.split("\\")[-1]
        if letter == "cat":
            labels = np.append(labels, n_img * [0])
        else:
            labels = np.append(labels, n_img * [1])
    temp = np.array([images, labels])
    temp = temp.transpose()  # 转置
    np.random.shuffle(temp)  # 顺序打乱，多维矩阵中，只对第一维（行）做打乱顺序操作
    image_list = list(temp[:, 0])
    label_list = list(temp[:, 1])
    label_list = [int(float(i)) for i in label_list]
    return image_list, label_list

    """获取一批次的数据"""


def get_batch(image_list, label_list, img_width, img_height, batch_size, capacity):
    image = tf.cast(image_list, tf.string)
    label = tf.cast(label_list, tf.int32)
    """tf.train.slice_input_producer([image,label],num_epochs=10),随机产生一个图片和标签,num_epochs=10,
    则表示把所有的数据过10遍，使用完所有的图片数据为一个epoch,这是重复使用10次。上面的用法表示你的数据集和标签
    已经全部加载到内存中了，如果数据集非常庞大，我们通过这个函数也可以只加载图片的路径，放入图片的path"""
    input_queue = tf.train.slice_input_producer([image, label])

    label = input_queue[1]
    image_contents = tf.read_file(input_queue[0])  # input_queue[0]图片路径，读取图片
    image = tf.image.decode_jpeg(image_contents, channels=3)
    """
    tf.image.resize_images：将原始图像缩放成指定的图像大小，其中的参数method（默认值为ResizeMethod.BILINEAR）提供了四种插值算法，具体解释可以参考图像几何变换（缩放、旋转）中的常用的插值算法 
    tf.image.resize_image_with_crop_or_pad：剪裁或填充处理，会根据原图像的尺寸和指定的目标图像的尺寸选择剪裁还是填充，如果原图像尺寸大于目标图像尺寸，则在中心位置剪裁，反之则用黑色像素填充。 
    tf.image.central_crop：比例调整，central_fraction决定了要指定的比例，取值范围为(0，1]，该函数会以中心点作为基准，选择整幅图中的指定比例的图像作为新的图像。
    """
    image = tf.image.resize_image_with_crop_or_pad(image=image, target_width=img_width, target_height=img_height)
    image = tf.image.per_image_standardization(image)  # 将图片标准化
    image_batch, label_batch = tf.train.batch(tensors=[image, label], batch_size=batch_size, num_threads=64,
                                              capacity=capacity)
    label_batch = tf.reshape(label_batch, [batch_size])
    return image_batch, label_batch


def rebuild(dir):
    """图片重新设置大小，统一为227*227的图片"""
    for root, dirs, files in os.walk(dir):
        for file in files:
            filepath = os.path.join(root, file)
            try:
                image = cv2.imread(filepath)
                dim = (227, 227)
                resized = cv2.resize(image, dim)
                path = "./data/" + file
                cv2.imwrite(path, resized)
            except:
                print(filepath)
                os.remove(filepath)
    cv2.waitKey(0)


def int64_feature(value):
    return tf.train.Feature(bytes_list=tf.train.Int64List(value=[value]))


def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def convert_to_tfrecord(images_list, label_list, save_dir, name):
    filename = os.path.join(save_dir, name + ".tfrecords")
    n_samples = len(label_list)
    writer = tf.python_io.TFRecordWriter(filename)
    print("\n Transform start........")
    for i in np.arange(0, n_samples):
        try:
            image = io.imread(images_list[i])  # image 必须是一个数组
            image_raw = image.tostring()
            label = int(label_list[i])
            example = tf.train.Example(features=tf.train.Features(feature={
                "label": int64_feature(label),
                "image_raw": bytes_feature(image_raw)
            }))
            writer.write(example.SerializeToString())
        except IOError as e:
            print("Could not read", images_list[i])
        writer.close();
        print("Transform done!")


"""
读取tfrecords记录，选取batch_size条记录，格式化为原来的格式
"""


def read_and_decode(tfrecords_file, batch_size):
    filename_queue = tf.train.string_input_producer([tfrecords_file])

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    img_features = tf.parse_single_example(
        serialized=serialized_example,
        features={
            "label": tf.FixedLenFeature([], tf.int64),
            "image_raw": tf.FixedLenFeature([], tf.string)
        }
    )
    image = tf.decode_raw(img_features["image_raw", tf.uint8])
    image = tf.reshape(image, [227, 227, 3])
    label = tf.cast(img_features["label"], tf.int32)
    image_batch, label_batch = tf.train.shuffle_batch(
        [image, label],
        batch_size=batch_size,
        min_after_dequeue=100,
        num_threads=64,
        capacity=200
    )
    return image_batch, tf.reshape(label_batch, [batch_size])

def onehot(labels):
    n_sample=len(labels)
    n_class=max(labels)+1
    onehot_labels=np.zeros((n_sample,n_class))
    onehot_labels[np.arange(n_sample),labels]=1
    return  onehot_labels





































