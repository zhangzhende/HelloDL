import torch as t

"""
这里主要介绍pytorch的基本
"""

"""
Tensor 部分
"""
def function1():
    # 构建5*3的矩阵
    x = t.Tensor(5, 3)
    print(x)


# function1()

def function2():
    # 使用【0,1】的均匀分布的随机初始化二维矩阵
    x = t.rand(5, 3)
    print(x)
    # 查看x的形状
    print(x.size())
    # 查看列的个数
    print(x.size()[1], x.size(1))


# function2()

x = t.rand(5, 3)
y = t.rand(5, 3)


#TODO 注意：函数名后面带下划线_的函数会修改Tensor本身，例如y.add_(x)，不带_的不会



def function3():
    """
    加法的表示
    :return: 
    """
    print(x)
    print(y)
    z1 = x + y
    print("z1:", z1)
    z2 = t.add(x, y)
    print("z2:", z2)
    # 不改变y的值，返回相加额结果
    z3 = y.add(x)
    # 改变y的值，返回x+y的返回结果
    y.add_(x)


# function3()

import numpy as np


def function4():
    # 新建一个全是1的tensor
    a = t.ones(5)
    print(a)
    # tensor---->numpy
    b = a.numpy()
    print(b)
    print("==========================")
    c = np.ones(5)
    # numpy------>tensor
    d = t.from_numpy(c)
    print(c)
    print(d)
    # tensor和numpy对象共享内存，相互转换快，但是，一旦一个改变，另一个也会随之改变
    d.add_(1)
    print("===========================")
    print(c)
    print(d)


# function4()



def function5():
    x = t.rand(5, 3)
    y = t.rand(5, 3)
    print(t.cuda.is_available())
    # tensor 可通过.cuda() 方法转化为GPU的tensor
    if t.cuda.is_available():
        x = x.cuda()
        y = y.cuda()
        z = x + y


# function5()

"""
Variable 变量部分
"""
#TODO 注意grad在反向传播过程中是累加的，也就是说对于第二次累加会加上之前的值
from torch.autograd import Variable
def function6():
    x=Variable(t.ones(2,2),requires_grad=True)
    print(x)
    #    y=x.sum()=(x[0][0]+x[0][1]+x[1][0]+x[1][1])
    y=x.sum()
    print(y)
    print(y.grad_fn)
    y.backward()
    #每个值得梯度都是1
    print(x.grad)
    #梯度值清0
    x=x.grad.data.zero_()
    print(x)
function6()
