from PIL import Image
from torchvision.transforms import  ToTensor,ToPILImage
import torch as t
from torch import nn
from torch.autograd import Variable as V
import matplotlib.pyplot as plt

"""
常用的神经网路------------------图像相关层
"""
to_tensor=ToTensor()
to_pil=ToPILImage()
bathpath="../data/image/17.jpg"
lena=Image.open(bathpath)

#TODO 卷积层
def function1():
    input=to_tensor(lena).unsqueeze(0)

    #锐化卷积核
    kernel=t.ones(3,3)/-9
    kernel[1][1]=1
    conv=nn.Conv2d(in_channels=3,out_channels=1,kernel_size=(3,3),stride=1,bias=False)
    conv.weight.data=kernel.view(1,3,1,3)
    out =conv(V(input))
    image=to_pil(out.data.squeeze(0))
    plt.imshow(image)
    plt.show()

#TODO 池化层
def function2():
    input = to_tensor(lena).unsqueeze(0)
    pool=nn.AvgPool2d(kernel_size=2,stride=2)
    out =pool(V(input))
    image=to_pil(out.data.squeeze(0))
    plt.imshow(image)
    plt.show()

#TODO 全连接层
def function3():
    input=V(t.randn(2,3))
    linear=nn.Linear(3,4)
    h=linear(input)
    print(h)
    """
    tensor([[ 2.1188, -0.5910, -1.1972, -0.8348],
        [-1.3983,  0.6351,  0.7712,  0.4022]])
    """


#TODO 统一规范化层
def function4():
    input = V(t.randn(2, 3))
    linear = nn.Linear(3, 4)
    h = linear(input)
    #4channel  初始化标准误差为4，均值为0
    bn=nn.BatchNorm1d(4)
    bn.weight.data=t.ones(4)*4
    bn.bias.data=t.zeros(4)
    bn_out=bn(h)
    #注意输出的均值和方差，方差是标准差的平方，计算无偏方差分母会减1，使用unbiased=False，分母不减1
    print(bn_out.mean(0),bn_out.var(0,unbiased=False))
    """
    tensor(1.00000e-06 *
       [ 0.2384,  0.0000, -8.1062,  0.0000]) tensor([ 15.9943,  15.9840,  15.2597,  15.9456])
    """

#TODO dropout层，防止过拟合
def function5():
    input = V(t.randn(2, 3))
    linear = nn.Linear(3, 4)
    h = linear(input)
    # 4channel  初始化标准误差为4，均值为0
    bn = nn.BatchNorm1d(4)
    bn.weight.data = t.ones(4) * 4
    bn.bias.data = t.zeros(4)
    bn_out = bn(h)

    dropout=nn.Dropout(0.5)
    o=dropout(bn_out)
    print(o)
    """
    tensor([[ 7.9514, -7.9999, -0.0000, -7.9999],
        [-7.9514,  0.0000,  0.0000,  0.0000]])
    """

"""
常用的神经网络层----------------激活函数
"""

#TODO RELU激活函数
def function6():
    relu=nn.ReLU(inplace=True)
    input =V(t.randn(2,3))
    print(input)
    output=relu(input)#小于0的都被截断为0
    print(output)
    """
    tensor([[ 0.2586,  0.6939, -0.6228],
        [-0.6443,  1.5716,  0.0142]])
    tensor([[ 0.2586,  0.6939,  0.0000],
        [ 0.0000,  1.5716,  0.0142]])
    """

#TODO Sequential 的三种写法
def function7():
    net1=nn.Sequential()
    net1.add_module("conv",nn.Conv2d(3,3,3))
    net1.add_module("batchnorm",nn.BatchNorm2d(3))
    net1.add_module("activation_layer",nn.ReLU())

    net2=nn.Sequential(
        nn.Conv2d(3,3,3),
        nn.BatchNorm2d(3),
        nn.ReLU()
    )

    from collections import OrderedDict
    net3=nn.Sequential(OrderedDict([
        ("conv1",nn.Conv2d(3,3,3)),
        ("bn1",nn.BatchNorm2d(3)),
        ("relu1",nn.ReLU())
    ]))
    print("net1:",net1)
    print("net2:",net2)
    print("net3:",net3)
    """
    net1: Sequential(
  (conv): Conv2d(3, 3, kernel_size=(3, 3), stride=(1, 1))
  (batchnorm): BatchNorm2d(3, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (activation_layer): ReLU()
)
net2: Sequential(
  (0): Conv2d(3, 3, kernel_size=(3, 3), stride=(1, 1))
  (1): BatchNorm2d(3, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (2): ReLU()
)
net3: Sequential(
  (conv1): Conv2d(3, 3, kernel_size=(3, 3), stride=(1, 1))
  (bn1): BatchNorm2d(3, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (relu1): ReLU()
)
    """

    #可根据名字取出字module
    print("net1.conv:",net1.conv,"\nnet2[0]:",net2[0],"\nnet3.conv1:",net3.conv1)
    """
    net1.conv: Conv2d(3, 3, kernel_size=(3, 3), stride=(1, 1)) 
    net2[0]: Conv2d(3, 3, kernel_size=(3, 3), stride=(1, 1)) 
    net3.conv1: Conv2d(3, 3, kernel_size=(3, 3), stride=(1, 1))
    """
    input =V(t.rand(1,3,4,4))
    output=net1(input)
    output=net2(input)
    output=net3(input)
    output=net3.relu1(net1.batchnorm(net1.conv(input)))

    modellist=nn.ModuleList([nn.Linear(3,4),nn.ReLU(),nn.Linear(4,2)])
    input=V(t.randn(1,3))
    for model in modellist:
        input=model(input)

#TODO ModuleList 是Module的子类，当在Module中使用他时，就能自动识别为字Module
class MyModule(nn.Module):

    def __init__(self):
        super(MyModule, self).__init__()
        self.list=[nn.Linear(3,4),nn.ReLU()]
        self.module_list=nn.ModuleList([nn.Conv2d(3,3,3),nn.ReLU()])

    def forward(self, *input):
        pass

#TODO 测试 ModuleList 是Module的子类，当在Module中使用他时，就能自动识别为字Module
def function8():
    model=MyModule()
    print(model)
    for name,param in model.named_parameters():
        print(name,param.size())

    """
    从输出中可以看出，使用list时，其中的子Module不会主Module识别，而ModuleList中的子Module能够被主module识别
MyModule(
  (module_list): ModuleList(
    (0): Conv2d(3, 3, kernel_size=(3, 3), stride=(1, 1))
    (1): ReLU()
  )
)
module_list.0.weight torch.Size([3, 3, 3, 3])
module_list.0.bias torch.Size([3])
    """

"""
常用的神经网络层----------------循环神经网络
"""
#TODO LSTM 使用
def function9():
    t.manual_seed(1000)
    #batch_size=3,序列长度为2，序列中每个元素占4维
    input=V(t.randn(2,3,4))
    #lstm输入向量4维，3个隐藏单元，1层
    lstm=nn.LSTM(4,3,1)
    #初始状态：1层，batch_size=3,3个隐藏单元
    h0=V(t.randn(1,3,3))
    c0=V(t.randn(1,3,3))
    out,hn=lstm(input,(h0,c0))
    print(out)
    """
    tensor([[[-0.3610, -0.1643,  0.1631],
         [-0.0613, -0.4937, -0.1642],
         [ 0.5080, -0.4175,  0.2502]],

        [[-0.0703, -0.0393, -0.0429],
         [ 0.2085, -0.3005, -0.2686],
         [ 0.1482, -0.4728,  0.1425]]])
    """

#TODO LSTMCELL使用
def function10():
    t.manual_seed(1000)
    input=V(t.randn(2,3,4))
    lstm=nn.LSTMCell(4,3)
    hx=V(t.randn(3,3))
    cx=V(t.randn(3,3))
    out=[]
    for i in input:
        hx,cx=lstm(i,(hx,cx))
        out.append(hx)
    t.stack(out)#维度叠加

#TODO Embedding层
def function11():
    embedding=nn.Embedding(4,5)
    #可以用预训练号的词向量初始化embedding
    embedding.weight.data=t.arange(0,20).view(4,5)
    input=V(t.arange(3,0,-1)).long()
    output=embedding(input)
    print(output)
"""
tensor([[ 15.,  16.,  17.,  18.,  19.],
        [ 10.,  11.,  12.,  13.,  14.],
        [  5.,   6.,   7.,   8.,   9.]])
"""

"""
常用的神经网络层----------------损失函数
"""
#TODO 交叉熵损失函数
def function12():
    #batch_size=3,计算对应每个类别的分数（只有两个类别）
    score=V(t.randn(3,2))
    #三个样本分别属于1,0,1类，label必须为LongTensor
    label=V(t.Tensor([1,0,1])).long()

    #loss 与普通的layer无差异
    criterion=nn.CrossEntropyLoss()
    loss=criterion(score,label)
    print(loss)
    """
    tensor(1.0347)
    """

if __name__=="__main__":
    #12
    function12()
    #11 embedding
    # function11()

    #10 lstmcell
    # function10()

    #9 lstm
    # function9()
    #3
    # function8()

    #Sequential 的三种写法
    # function7()

    #1relu激活函数
    # function6()

    #5 dropout
    # function5()

    #4统一规范化层
    # function4()

    #3 全连接层
    # function3()

    #2 池化层
    # function2()

    #1二维卷积层
    # function1()
