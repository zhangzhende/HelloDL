
import torch.nn as nn
import torch.nn.functional as Fun
import torch.optim as optim  #优化器
from torch.autograd import Variable  #变量
import torch  as t
import torchvision as tv
import torchvision.transforms as transforms
from torchvision.transforms import ToPILImage
import matplotlib.pyplot as plt
show=ToPILImage()#可以把Tensor转换成Image，方便可视化


"""
小试牛刀---CIFAR-10分类
"""

#TODO 准备数据，下载和加载loader
def prepareData():
    transform=transforms.Compose([
        transforms.ToTensor(),#转化为Tensor
        transforms.Normalize(mean=(0.5,0.5,0.5),std=(0.5,0.5,0.5))#归一化
    ])
    #训练集
    bathPath="../../../MyDeepLearning/data/cifar10/"
    trainset=tv.datasets.CIFAR10(
        root=bathPath,
        train=True,
        download=True,
        transform=transform
    )
    """
    class torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, sampler=None, batch_sampler=None, num_workers=0, collate_fn=<function default_collate at 0x4316c08>, pin_memory=False, drop_last=False)[source]
    数据加载器. 组合数据集和采样器,并在数据集上提供单进程或多进程迭代器.
    Parameters:
    dataset (Dataset) – 从该数据集中加载数据.
    batch_size (int, optional) – 每个 batch 加载多少个样本 (默认值: 1).
    shuffle (bool, optional) – 设置为 True 时, 会在每个 epoch 重新打乱数据 (默认值: False).
    sampler (Sampler, optional) – 定义从数据集中提取样本的策略. 如果指定, shuffle 值必须为 False.
    batch_sampler (Sampler, optional) – 与 sampler 相似, 但一次返回一批指标. 与 batch_size, shuffle, sampler, and drop_last 互斥.
    num_workers (int, optional) – 用多少个子进程加载数据. 0表示数据将在主进程中加载 (默认值: 0)
    collate_fn (callable, optional) – 合并样本列表以形成一个 mini-batch.
    pin_memory (bool, optional) – 如果为 True, 数据加载器会将张量复制到 CUDA 固定内存中, 然后再返回它们.
    drop_last (bool, optional) – 设定为 True 以丢掉最后一个不完整的 batch, 如果数据集大小不能被 batch size整除. 设定为 False 并且数据集的大小不能被 batch size整除, 则最后一个 batch 将会更小. (default: False)
    """
    trainLoader=t.utils.data.DataLoader(
        trainset,
        batch_size=4,
        shuffle=True,
        num_workers=2
    )
    #测试集
    testset=tv.datasets.CIFAR10(
        root=bathPath,
        train=False,
        download=True,
        transform=transform
    )
    testLoader=t.utils.data.DataLoader(
        dataset=testset,
        batch_size=4,
        shuffle=False,
        num_workers=2
    )
    classes=("plane","car","bird","cat","deer","dog","frog","horse","ship","truck")
    (data,label)=trainset[100]
    print(classes[label])
    #(data+1)/2 是为了还原被归一化的数据
    images=show((data+1)/2).resize((100,100))
    #plt.imshow()函数负责对图像进行处理，并显示其格式，而plt.show()则是将plt.imshow()处理后的函数显示出来
    # plt.imshow(images)
    # plt.show()
    return trainLoader,testLoader,classes

#TODO dataloader 输出展示数据
def function1():
    transform = transforms.Compose([
        transforms.ToTensor(),  # 转化为Tensor
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))  # 归一化
    ])
    # 训练集
    bathPath = "../../../MyDeepLearning/data/cifar10/"
    trainset = tv.datasets.CIFAR10(
        root=bathPath,
        train=True,
        download=True,
        transform=transform
    )

    trainLoader = t.utils.data.DataLoader(
        dataset=trainset,
        batch_size=4,
        shuffle=True,
        num_workers=2
    )
    # 测试集
    testset = tv.datasets.CIFAR10(
        root=bathPath,
        train=False,
        download=True,
        transform=transform
    )
    testLoader = t.utils.data.DataLoader(
        dataset=testset,
        batch_size=4,
        shuffle=False,
        num_workers=2
    )
    classes = ("plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")
    dataiter=iter(trainLoader)
    images,labels=dataiter.next()
    print("".join("%11s"%classes[labels[j]] for j in range(4)))
    image=tv.utils.make_grid((images+1)/2)
    image=show(image).resize((400,100))
    plt.imshow(image)
    plt.show()

#TODO 定义网络
class Net(nn.Module):
    """
    定义网络时需要继承nn,Module，并实现他的forward方法
    """

    def __init__(self):
        # nn.Module子类必须在构造函数中执行父类的构造函数
        # 下式等价于nn.Module.__init__(self)
        super(Net, self).__init__()
        #
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        # 仿射层/全连接层，y=Wx+b
        self.fc1 = nn.Linear(in_features=16 * 5 * 5, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=84)
        self.fc3 = nn.Linear(in_features=84, out_features=10)

    def forward(self, x):
        # 卷积--》激活--》池化
        x = Fun.max_pool2d(input=Fun.relu(input=self.conv1(x)), kernel_size=(2, 2))
        x = Fun.max_pool2d(input=Fun.relu(input=self.conv2(x)), kernel_size=2)
        # reshape,-1表示自适应
        x = x.view(x.size()[0], -1)
        x = Fun.relu(self.fc1(x))
        x = Fun.relu(self.fc2(x))
        x = Fun.relu(self.fc3(x))
        return x

#TODO 测试net，训练net
def function3():
    net =Net()
    # print(net)
    criterion=nn.CrossEntropyLoss()#交叉熵损失函数
    optimizer=optim.SGD(params=net.parameters(),lr=0.001,momentum=0.9)
    trainLoader,_,__=prepareData()
    for epoch in range(2):
        running_loss=0.0
        for i,data in enumerate(trainLoader,0):
            #输入数据
            inputs ,labels=data
            inputs,labels=Variable(inputs),Variable(labels)
            #梯度清0
            optimizer.zero_grad()

            outputs=net(inputs)
            loss=criterion(outputs,labels)
            loss.backward()
            #更新参数
            optimizer.step()
            #打印log信息,将pytorch更新到0.4.0最新版后对0.3.1版本代码会有如下警告，它在提醒用户下个版本这将成为一个错误,改成item()的方式
            # invalid index of a 0-dim tensor. This will be an error in PyTorch 0.5. Use tensor.item() to convert a 0-dim tensor to a Python number
            # running_loss+=loss.data[0]
            running_loss += loss.item()
            if i%2000==1999:
                print("[%d,%5d] loss :%.3f" %(epoch+1,i+1,running_loss/2000))
                running_loss=0.0
    print("finished traing!!")

#TODO 展示一波测试的图片
def function4():
    _,testLoader,classes=prepareData()
    dataiter=iter(testLoader)
    images,labels=dataiter.next()
    print("实际的label:","".join("%08s"%classes[labels[j]] for j in range(4)))
    image = tv.utils.make_grid((images + 1) / 2)
    image = show(image).resize((400, 100))
    plt.imshow(image)
    plt.show()

#TODO 简单拿几张照片测试
def function5():
    trainLoader, testLoader, classes = prepareData()
    dataiter = iter(testLoader)
    images, labels = dataiter.next()
    net =Net()
    criterion=nn.CrossEntropyLoss()#交叉熵损失函数
    optimizer=optim.SGD(params=net.parameters(),lr=0.001,momentum=0.9)
    for epoch in range(2):
        running_loss=0.0
        for i,data in enumerate(trainLoader,0):
            #输入数据
            inputs ,labels=data
            inputs,labels=Variable(inputs),Variable(labels)
            #梯度清0
            optimizer.zero_grad()
            outputs=net(inputs)
            loss=criterion(outputs,labels)
            loss.backward()
            #更新参数
            optimizer.step()
            running_loss += loss.item()
            if i%2000==1999:
                print("[%d,%5d] loss :%.3f" %(epoch+1,i+1,running_loss/2000))
                running_loss=0.0
    outputs=net(Variable(images))
    _,predict=t.max(outputs,1)
    print("预测结果：","".join("%5s"%classes[predict[j]] for j in range(4)))

# TODO 测试集上所有测试
def function6():
    trainLoader, testLoader, classes = prepareData()
    dataiter = iter(testLoader)
    images, labels = dataiter.next()
    net = Net()
    criterion = nn.CrossEntropyLoss()  # 交叉熵损失函数
    optimizer = optim.SGD(params=net.parameters(), lr=0.001, momentum=0.9)
    for epoch in range(2):
        running_loss = 0.0
        for i, data in enumerate(trainLoader, 0):
            # 输入数据
            inputs, labels = data
            inputs, labels = Variable(inputs), Variable(labels)
            # 梯度清0
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            # 更新参数
            optimizer.step()
            running_loss += loss.item()
            if i % 2000 == 1999:
                print("[%d,%5d] loss :%.3f" % (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0
    correct=0#预测正确的图片
    total=0#总图片数
    for data in testLoader:
        images,labels =data
        outputs = net(Variable(images))
        _, predict = t.max(outputs, 1)
        total+=labels.size(0)
        correct+=(predict==labels).sum()
    print("10000张测试集中的图片准确率为%d %%"%(100*correct/total))

#TODO GPU上训练,虽然本地电脑暂时不支持
def function7():
    trainLoader, testLoader, classes = prepareData()
    dataiter = iter(testLoader)
    images, labels = dataiter.next()
    net = Net()
    #GPU
    if t.cuda.is_available():
        net.cude()
        images=images.cuda()
        labels=labels.cuda()
    criterion = nn.CrossEntropyLoss()  # 交叉熵损失函数
    optimizer = optim.SGD(params=net.parameters(), lr=0.001, momentum=0.9)
    for epoch in range(2):
        running_loss = 0.0
        for i, data in enumerate(trainLoader, 0):
            # 输入数据
            inputs, labels = data
            inputs, labels = Variable(inputs), Variable(labels)
            # 梯度清0
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            # 更新参数
            optimizer.step()
            running_loss += loss.item()
            if i % 2000 == 1999:
                print("[%d,%5d] loss :%.3f" % (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0
    outputs = net(Variable(images))
    _, predict = t.max(outputs, 1)
    print("预测结果：", "".join("%5s" % classes[predict[j]] for j in range(4)))
if __name__ == '__main__':
    #1
    # prepareData()
    """
     The "freeze_support()" line can be omitted if the program
        is not going to be frozen to produce an executable.
    exitcode = _main(fd)
    少了下面这句会报错
    """
    #2
    # t.multiprocessing.freeze_support()
    # function1()
    #3
    # t.multiprocessing.freeze_support()
    # function3()
    #4
    # function4()
    #5
    # function5()
    function6()