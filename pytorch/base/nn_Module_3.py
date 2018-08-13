import torch as t
from torch import nn
from torch import optim
from torch.autograd import Variable as V

"""
优化器
"""


class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.features=nn.Sequential(
            nn.Conv2d(3,6,5),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(6,16,5),
            nn.ReLU(),
            nn.MaxPool2d(2,2)
        )
        self.classifier=nn.Sequential(
            nn.Linear(16*5*5,120),
            nn.ReLU(),
            nn.Linear(120,84),
            nn.ReLU(),
            nn.Linear(84,10)
        )
    def forward(self, x):
        x=self.features(x)
        x=x.view(-1,16*5*5)
        x=self.classifier(x)
        return x

#TODO 一般使用优化器
def function1():
    net=Net()
    optimizer=optim.SGD(params=net.parameters(),lr=1)
    optimizer.zero_grad()#梯度清零

    input=V(t.randn(1,3,32,32))
    output=net(input)
    output.backward(output)

    optimizer.step()

#TODO 为不同的子网络设置不同的学习率，在finetune中经常用到
#如果对某个参数不指定学习率就使用默认的学习率
def function2():
    net = Net()
    optimizer=optim.SGD([
        {"params":net.features.parameters()},#学习率为1e-5
        {"params":net.classifier.parameters(),"lr":1e-2}
    ],lr=1e-5)

if __name__=="__main__":
    #1
    function1()