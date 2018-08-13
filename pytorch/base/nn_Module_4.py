import torch as t
from torch import nn
from torch import optim
from torch.autograd import Variable as V

"""
nn.functional
和nn.Module中的layer对应，它更像一个方法，用来直接调用返回结果。
如果说模型有参数要学习，用Module，否则都可以用
"""

def function1():
    input =V(t.randn(2,3))
    model=nn.Linear(3,4)
    output1=model(input)
    output2=nn.functional.linear(input,model.weight,model.bias)
    output3=nn.functional.relu(input)
    output4=nn.ReLU()(input)
    print(output1==output2)
    print("output1:",output1)
    print("output2:",output2)
    print("output3:",output3)
    print("output4:",output4)
    """
tensor([[ 1,  1,  1,  1],
        [ 1,  1,  1,  1]], dtype=torch.uint8)
output1: tensor([[-0.2798, -0.4294,  1.5282,  1.2741],
        [-0.5852,  0.2249,  1.3691,  1.0810]])
output2: tensor([[-0.2798, -0.4294,  1.5282,  1.2741],
        [-0.5852,  0.2249,  1.3691,  1.0810]])
output3: tensor([[ 0.0000,  0.0000,  0.0000],
        [ 0.0000,  0.8476,  0.0000]])
output4: tensor([[ 0.0000,  0.0000,  0.0000],
        [ 0.0000,  0.8476,  0.0000]])
    """


#TODO 创建模型时，对于没有可学习参数的网络层【Relu,sigmod,tanh,池化等】可以用nn.functional代替
from torch.nn import functional as F
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1=nn.Conv2d(3,6,5)
        self.conv2=nn.Conv2d(6,16,5)
        self.fc1=nn.Linear(16*5*5,120)
        self.fc2=nn.Linear(120,84)
        self.fc3=nn.Linear(84,10)

    def forward(self, x):
        #avg_pool2d(input, kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True)
        x=F.avg_pool2d(F.relu(self.conv1(x)),2)
        x=F.avg_pool2d(F.relu(self.conv2(x)),2)
        x=x.view(-1,16*5*5)
        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        x=self.fc3(x)
        return x




if __name__=="__main__":
    function1()
