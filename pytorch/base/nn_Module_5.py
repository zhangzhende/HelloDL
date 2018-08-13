import torch as t
from torch import nn
from torch.nn import init

"""
初始化策略
"""


#TODO 利用nn.init 初始化
def function1():
    linear=nn.Linear(3,4)
    t.manual_seed(1)
    #等价于  linear.weight.data.normal_(0,std)
    #nn.init.xavier_normal is now deprecated in favor of nn.init.xavier_normal_.init.xavier_normal(linear.weight)
    init.xavier_normal_(linear.weight)
    print(linear.weight)
    """
    Parameter containing:
    tensor([[ 0.3535,  0.1427,  0.0330],
        [ 0.3321, -0.2416, -0.0888],
        [-0.8140,  0.2040, -0.5493],
        [-0.3010, -0.4769, -0.0311]])
    """

#TODO 直接初始化
def function2():
    import math
    linear = nn.Linear(3, 4)
    t.manual_seed(1)

    std=math.sqrt(2)/math.sqrt(7)
    linear.weight.data.normal_(0,std)
    print(linear.weight)
    """
    Parameter containing:
    tensor([[ 0.3535,  0.1427,  0.0330],
        [ 0.3321, -0.2416, -0.0888],
        [-0.8140,  0.2040, -0.5493],
        [-0.3010, -0.4769, -0.0311]])
    """

if __name__=="__main__":

    #2
    function2()

    #1
    # function1()