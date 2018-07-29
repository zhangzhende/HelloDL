from MyDeepLearning.Common.datasets import flowers
# from myProject.tensorflowDemo.slimAdvance import configure
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.python.slim.data.dataset_data_provider as provider
import MyDeepLearning.Common.datasets.inception_resnet_v2 as V2model
import MyDeepLearning.Common.datasets.inception_resnet_v2_modify as V2modelModify
import os

"""
Slim的使用进阶

使用slim预训练模型进行Finetuning
Inception-ResNet-V2模型
由inception-v3模型演变而来，主要是增加了大量的残差网络
"""
inceptionV2ModelPath="../../MyDeepLearning/data/InceptionV2ResnetV2"
inceptionV2ModelCkName="inception_resnet_v2_2016_08_30.ckpt"
flowersDatDir="../../MyDeepLearning/data/flowers/"
def loadModel():
    """
    载入预训练模型
    :return: 
    """
    checkpointDir=inceptionV2ModelPath
    with tf.Graph().as_default():
        img=tf.random_normal([1,299,299,3])
        with slim.arg_scope(V2model.inception_resnet_v2_arg_scope()):
            pre,_=V2model.inception_resnet_v2(inputs=img,num_classes=1001,is_training=False)

            modelPath=os.path.join(checkpointDir,"inception_resnet_v2_2016_08_30.ckpt")
            variables=slim.get_model_variables("InceptionResnetV2")
            init_fn=slim.assign_from_checkpoint_fn(model_path=modelPath,var_list=variables)
            with tf.Session() as sess:
                init_fn(sess)
                print((sess.run(pre)))
                print("done!")
#[[-0.83881414 -1.9929581  -0.280567   ... -2.05064    -0.20870075 0.7230984 ]]
#done!
loadModel()

def loadBatch(dataset,batchSize=4,height=299,width=299):
    """
    批量加载数据
    :param dataset: 
    :param batchSize: 
    :param height: 
    :param width: 
    :return: 
    """
    dataProvider=provider.DatasetDataProvider(dataset=dataset,common_queue_capacity=8,common_queue_min=1)
    imageRaw,label=dataProvider.get(["image","label"])
    imageRaw=tf.image.resize_images(images=imageRaw,size=[height,width])
    imageRaw=tf.image.convert_image_dtype(image=imageRaw,dtype=tf.float32)
    imageRaw,labels=tf.train.batch(
        tensors=[imageRaw,label],
        batch_size=batchSize,
        num_threads=1,
        capacity=2*batchSize
    )
    return imageRaw,labels
"""
20-7
"""

def trainModel():
    """
    不修改最终的分类【1001】情况下修改自己的输出
    :return: 
    """
    fintuing=tf.Graph()
    with fintuing.as_default():
        tf.logging.set_verbosity(tf.logging.INFO)
        #获取数据集
        dataset=flowers.get_split("train",flowersDatDir)
        images,labels=loadBatch(dataset=dataset)

        #载入模型，此时模型还没有载入参数
        with slim.arg_scope(V2model.inception_resnet_v2_arg_scope()):
            logits, end_points=V2model.inception_resnet_v2(inputs=images,num_classes=1001)
            prob=tf.nn.softmax(logits)

            oneHotLabels=slim.one_hot_encoding(labels=labels,num_classes=1001)

            #创建损失函数
            slim.losses.softmax_cross_entropy(logits=prob,onehot_labels=oneHotLabels)
            totalLoss=slim.losses.get_total_loss()

            #创建训练节点
            optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.01)
            trainOp=slim.learning.create_train_op(total_loss=totalLoss,optimizer=optimizer)

            #准备载入模型权重的函数
            modelPath=os.path.join(inceptionV2ModelPath,inceptionV2ModelCkName)
            variables=slim.get_model_variables("InceptionResnetV2")
            init_fn=slim.assign_from_checkpoint_fn(model_path=modelPath,var_list=variables)
            #正式载入模型并开始训练
            with tf.Session() as sess:
                init_fn(sess)
                print("done!")
            final_loss=slim.learning.train(
                train_op=trainOp,
                logdir=None,
                number_of_steps=10#训练10波
            )
# trainModel()
"""
20-8
"""
def trainModelWithMyClass():
    """
    修改最终的输出分类【1001】==》【5】,有1001个分类改为5个分类
    :return: 
    """
    fintuing = tf.Graph()
    with fintuing.as_default():
        tf.logging.set_verbosity(tf.logging.INFO)
        # 获取数据集
        dataset = flowers.get_split("train", flowersDatDir)
        images, labels = loadBatch(dataset=dataset)

        # 载入模型，此时模型还没有载入参数
        with slim.arg_scope(V2model.inception_resnet_v2_arg_scope()):
            logits, end_points = V2model.inception_resnet_v2(inputs=images, num_classes=1001)
            #增加一个全连接层，将输出1001分类改为输出5个分类
            logits=slim.fully_connected(inputs=logits,num_outputs=5)
            prob = tf.nn.softmax(logits)

            oneHotLabels = slim.one_hot_encoding(labels=labels, num_classes=5)

            # 创建损失函数
            slim.losses.softmax_cross_entropy(logits=prob, onehot_labels=oneHotLabels)
            totalLoss = slim.losses.get_total_loss()

            # 创建训练节点
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
            trainOp = slim.learning.create_train_op(total_loss=totalLoss, optimizer=optimizer)

            # 准备载入模型权重的函数
            modelPath = os.path.join(inceptionV2ModelPath, inceptionV2ModelCkName)
            variables = slim.get_model_variables("InceptionResnetV2")
            init_fn = slim.assign_from_checkpoint_fn(model_path=modelPath, var_list=variables)
            # 正式载入模型并开始训练
            with tf.Session() as sess:
                init_fn(sess)
                print("done!")
            final_loss = slim.learning.train(
                train_op=trainOp,
                logdir=None,
                number_of_steps=10  # 训练10波
            )
# trainModelWithMyClass()

"""
20-9
"""
def trainModelWithNew():
    """
    从头训练自己的模型
    :return: 
    """
    graph = tf.Graph()
    with graph.as_default():
        tf.logging.set_verbosity(tf.logging.INFO)
        # 获取数据集
        dataset = flowers.get_split("train", flowersDatDir)
        images, labels = loadBatch(dataset=dataset)

        # 载入模型，此时模型还没有载入参数
        with slim.arg_scope(V2model.inception_resnet_v2_arg_scope()):
            logits, end_points = V2model.inception_resnet_v2(inputs=images, num_classes=5)
            prob = tf.nn.softmax(logits)

            oneHotLabels = slim.one_hot_encoding(labels=labels, num_classes=5)

            print("oneHotLabels:",oneHotLabels.shape)
            print("prob:", prob.shape)
            # 创建损失函数
            slim.losses.softmax_cross_entropy(logits=prob, onehot_labels=oneHotLabels)
            totalLoss = slim.losses.get_total_loss()

            # 创建训练节点
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
            trainOp = slim.learning.create_train_op(total_loss=totalLoss, optimizer=optimizer)

            final_loss = slim.learning.train(
                train_op=trainOp,
                logdir=None,
                number_of_steps=10  # 训练10波
            )
            print("trianing done and finished")
# trainModelWithNew()

def getInitFn(sess):
    """
    
    :param sess: 
    :return: 
    """
    checkPoint_exclude_scopes=["InceptionResnetV2/AuxLogits","InceptionResnetV2/Logits"]
    exclusions=[scope.strip() for scope in checkPoint_exclude_scopes]

    variablesToRestore=[]
    for var in slim.get_model_variables():
        excluded=False
        for exclusion in exclusions:
            if var.op.name.startswith(exclusion):
                excluded=True
                break
        if not excluded:
            variablesToRestore.append(var)
    modelPath=os.path.join(inceptionV2ModelPath,inceptionV2ModelCkName)
    init_fn=slim.assign_from_checkpoint_fn(model_path=modelPath,var_list=variablesToRestore)
    return init_fn(sess)

"""
20-10
"""
def trainModelModify():
    """
    修改inception-resnet-v2的预训练模型输出层级
    :return: 
    """
    with tf.Graph().as_default():
        img=tf.random_normal([1,299,299,3])

        with slim.arg_scope(V2modelModify.inception_resnet_v2_arg_scope()):
            pre,_=V2modelModify.inception_resnet_v2(inputs=img,is_training=False)

        with tf.Session() as sess:
            init_fn=getInitFn(sess=sess)
            res=(sess.run(pre))
            print(res.shape)
# trainModelModify()

























