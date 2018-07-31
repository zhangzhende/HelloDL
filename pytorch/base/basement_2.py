import torch.nn as nn
import torch.nn.functional as Fun
from torch.autograd import Variable  #变量
import torch as t
import torch.optim as optim  #优化器

"""
torch.nn是专门为神经网络设计的接口模块
nn.Module的主要一些使用，套路
"""


# TODO 定义网络
class Net(nn.Module):
    """
    定义网络时需要继承nn,Module，并实现他的forward方法
    """

    def __init__(self):
        # nn.Module子类必须在构造函数中执行父类的构造函数
        # 下式等价于nn.Module.__init__(self)
        super(Net, self).__init__()
        #
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
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


net = Net()
# print(net)
"""
输出网络结构：
Net(
  (conv1): Conv2d(1, 5, kernel_size=(5, 5), stride=(1, 1))
  (conv2): Conv2d(6, 16, kernel_size=(5, 5), stride=(1, 1))
  (fc1): Linear(in_features=400, out_features=120, bias=True)
  (fc2): Linear(in_features=120, out_features=84, bias=True)
  (fc3): Linear(in_features=84, out_features=10, bias=True)
)
"""
params = list(net.parameters())
# print(len(params))
# 输出：10
for name, parameters in net.named_parameters():
    # print(name,":",parameters.size())
    pass
"""
输出参数大小size
conv1.weight : torch.Size([5, 1, 5, 5])
conv1.bias : torch.Size([5])
conv2.weight : torch.Size([16, 6, 5, 5])
conv2.bias : torch.Size([16])
fc1.weight : torch.Size([120, 400])
fc1.bias : torch.Size([120])
fc2.weight : torch.Size([84, 120])
fc2.bias : torch.Size([84])
fc3.weight : torch.Size([10, 84])
fc3.bias : torch.Size([10])
"""
# forward 函数的输入输出都是Variable，只有Variable才有自动求导，所以在输入时，tensor要转化为Variable
## 构建1*1*32*32的矩阵
input = Variable(t.randn(1, 1, 32, 32))
# nn.Conv2d的输入必须是4维的，nSamples,nChannels,Height,Width
out = net(input)
a = out.size()
# print(a)
# 输出：torch.Size([1, 10])
# net.zero_grad()#所有参数的梯度清0
# out.backward(Variable(t.ones(1,10)))#反向传播

# TODO 损失函数
# output=net(input)
# print(out)
#  输出：tensor([[ 0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9.]])
a = t.arange(0, 10)
# print(a)
# 输出：tensor([ 0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9.]) 【10】
target = Variable(a)
criterion = nn.MSELoss()  # 均方误差
# criterion=nn.CrossEntropyLoss()#交叉熵损失
# 书上这块有问题，out为1*10的，target为10，维度不一致，所以下面out去取第一个了
#todo 也可以这样target = target.reshape((1,10))
loss = criterion(out[0], target)
# print(loss)
# 输出： tensor(28.0211)

# TODO 运行.backward，观察调用之前和之后的grad
net.zero_grad()  # 清空
print("反向传播之前的conv1.bias的梯度")
print(net.conv1.bias.grad)
loss.backward()
print("反向传播之后的conv1.bias的梯度")
print(net.conv1.bias.grad)
"""
反向传播之前的conv1.bias的梯度
None
反向传播之后的conv1.bias的梯度
tensor(1.00000e-02 *
       [-6.1403,  0.4963,  5.7112,  3.5512,  6.0202, -1.6396])
"""

#TODO 优化器
optimizer =optim.SGD(params=net.parameters(),lr=0.01)#新建一个优化器，指定参数和学习率
#在训练过程中，先把梯度清0
optimizer.zero_grad()
#计算损失
output=net(input)[0]
loss=criterion(output,target)
loss.backward()
optimizer.step()
