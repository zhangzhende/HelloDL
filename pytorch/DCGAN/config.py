"""
生成对抗网络的配置信息
"""
class Config(object):
    data_path="./data/"    #数据集存放的路径
    num_workers=4#加载数据的进程数
    image_size=96#图片尺寸
    batch_size=256#
    max_epoch=200#
    lr1=2e-4#生成器的学习率
    lr2=2e-4#判别器的学习率
    beta1=0.5#Adam优化器的beta1参数
    use_gpu=False#是否使用GPU加速
    nz=100#噪声的维度
    ngf=96#生成器的feature map数
    ndf=96#判别器的feature map 数

    save_path="./imgs/"  #生成的图片存放路径

    vis=True #是否使用visdom可视化
    env="GAN" #环境env
    plot_every=20 #每隔20个batch画图一次

    debug_file="./tmp/debuggan/"#存在文件就行如debug模式
    d_every=1#每一个batch训练一次判别器
    g_every=5 #每5个batch训练一次生成器
    decay_every=10#每10个epoch保存一次模型
    netd_path="./checkpoints/netd_211.pth"#判别器预训练模型
    netg_path="./checkpoints/netg_211.pth"#生成器预训练模型

    #测试时使用的参数
    gan_image="result.png"
    gen_num=64
    gen_search_num=512
    gen_mean=0
    gen_std=1

opt=Config()

