import torch as t
from torch import nn
from torch import optim
from torch.autograd import Variable as V
from torch.nn import init

"""
nn.Module 深入分析
1.module属性的设置获取方式，多种多样
2.模型保存，加载
3.将module放在GPU上运算
"""

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        #等价于 self.register_parameter(name="param1",param=nn.Parameter(t.rand(3,3)))
        self.param1=nn.Parameter(t.rand(3,3))#参数设置
        self.submodel1=nn.Linear(3,4)#子模块设置

    def forward(self, input):
        x=self.param1@input
        x=self.submodel1(x)
        return x


def function1():
    net=Net()
    print("net:",net)
    print("net._modules:",net._modules)
    print("net._parameters:",net._parameters)
    print("net.param1:",net.param1)
    """
net: Net( (submodel1): Linear(in_features=3, out_features=4, bias=True))
net._modules: OrderedDict([('submodel1', Linear(in_features=3, out_features=4, bias=True))])
net._parameters: OrderedDict([('param1', Parameter containing:
tensor([[ 0.2444,  0.7476,  0.8715],
        [ 0.0503,  0.7686,  0.0292],
        [ 0.8084,  0.9973,  0.6383]]))])
net.param1: Parameter containing:
tensor([[ 0.2444,  0.7476,  0.8715],
        [ 0.0503,  0.7686,  0.0292],
        [ 0.8084,  0.9973,  0.6383]])
    """

#TODO 对于dropout训练和不训练的时候处理方式不一样，训练时dropout生效，不训练时，dropout不生效
def function2():
    input=V(t.arange(0,12).view(3,4))
    model=nn.Dropout()
    output=model(input)
    #在训练阶段，dropout会生效，一半左右会变成0，默认training=true
    print(output)
    """
    tensor([[  0.,   0.,   0.,   6.],
        [  8.,   0.,   0.,   0.],
        [  0.,  18.,   0.,   0.]])
    """
    model.training=False
    output2=model(input)
    print(output2)
    """
    tensor([[  0.,   1.,   2.,   3.],
        [  4.,   5.,   6.,   7.],
        [  8.,   9.,  10.,  11.]])
    """
#TODO batchnorm,dropout,instancenorm 在训练和测试阶段的处理方式差距较大，所以训练和测试的时候要指明工作环境，
# model.train()函数将所有model统一设置trianning=true,model.eval()统一设置所有的trainning=False。
def function3():
    net =Net()
    print(net.training,net.submodel1.training)
    net.eval()
    print(net.training, net.submodel1.training)
    net.train()
    print(net.training, net.submodel1.training)
    """
        True True
        False False
        True True
    """

"""
模型保存，加载
"""
#TODO 模型的保存和加载
def function4():
    net =Net()
    t.save(net.state_dict(),"../data/other/net.pth")
    net2=Net()
    net2.load_state_dict(t.load("../data/other/net.pth"))

#TODO 将module放在GPU上计算比较简单，module=module.cuda(),input=input.cuda()即可
def function5():
    """
    指定使用哪些GPU计算
    :return: 
    """
    input = V(t.arange(0, 12).view(3, 4))
    net=Net()
    #方法1
    new_net=nn.DataParallel(net,device_ids=[0,1])
    output=new_net(input)
    #方法2
    output2=nn.parallel.data_parallel(net,input,device_ids=[0,1])

if __name__=="__main__":

    #4
    function4()

    #3
    # function3()

    #2
    # function2()

    #1
    # function1()