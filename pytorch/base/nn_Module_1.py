import torch as t
from torch import nn
from torch.autograd import Variable as V

"""
nn.Module基本
"""

#TODO 使用nn.Module 实现自己的全连接层（仿射层）
class Linear(nn.Module):

    def __init__(self,in_features,out_features):
        super(Linear,self).__init__()#等价于nn.Module.__init__(self)
        self.w=nn.Parameter(t.randn(in_features,out_features))
        self.b=nn.Parameter(t.randn(out_features))

    def forward(self, x):
        x=x.mm(self.w)
        return x+self.b.expand_as(x)

#TODO 自定义全连接层网络的使用
def function1():
    layer=Linear(4,3)
    input=V(t.randn(2,4))
    output=layer(input)
    print(output)
    for name ,parameter in layer.named_parameters():
        print(name,parameter)
    """
    tensor([[-1.1369,  3.0402,  2.8769],
        [-3.5722,  2.0994,  4.5806]])
    w Parameter containing:
        tensor([[ 1.6384,  0.0210, -0.3826],
        [ 0.6096,  0.2238, -0.9096],
        [-0.1480, -0.9202, -0.0841],
        [-1.1664,  0.8315,  0.8430]])
    b Parameter containing:
        tensor([-1.7887,  0.9064,  2.6205])
    """

#TODO 多层感知机实现
class Perceptron(nn.Module):

    def __init__(self,in_features,hidden_features,out_features):
        nn.Module.__init__(self)#等价于 super(Perceptron,self).__init__()
        self.layer1=Linear(in_features,hidden_features)
        self.layer2=Linear(hidden_features,out_features)

    def forward(self,x):
        x=self.layer1(x)
        x=t.sigmoid(x)
        return self.layer2(x)

def function2():
    perceptron =Perceptron(3,4,1)
    for name,param in perceptron.named_parameters():
        print(name,param.size())
    """
    layer1.w torch.Size([3, 4])
    layer1.b torch.Size([4])
    layer2.w torch.Size([4, 1])
    layer2.b torch.Size([1])
    """



if __name__=="__main__":
    #查看多层感知机
    function2()
    #1使用自定义全连接层
    # function1()