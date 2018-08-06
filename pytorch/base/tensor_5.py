import torch as t
from matplotlib import pyplot as plt


"""
tensor 一些常用操作：
修改数据类型
逐元素操作：常见：abs/sqrt/div/exp/fmod/log/pow
归并操作，此类操作输入一般大于输出，如求和sum等
比较操作，如< ,>等
线性代数
线性回归小栗子
"""
# TODO ===========tensor数据类型相关======================

# TODO 不好用了
def function1():
    t.set_default_tensor_type('torch.IntTensor')  # 修改默认的tensor类型
    #    only floating-point types are supported as the default type
    a = t.Tensor(2, 3)
    print(a)

#TODO 修改数据类型
def function2():
    a = t.Tensor(2, 3)
    b = a.int()
    c = b.long()
    d = a.type_as(b)
    e = a.new(2, 3)
    print("b:", b)
    print("c:", c)
    print("d:", d)
    print("e:", e)
    """
    b: tensor([[-7.9231e+06,  0.0000e+00, -7.0000e+00],
        [ 0.0000e+00, -7.0000e+00,  0.0000e+00]], dtype=torch.int32)
    c: tensor([[-7.9231e+06,  0.0000e+00, -7.0000e+00],
        [ 0.0000e+00, -7.0000e+00,  0.0000e+00]])
    d: tensor([[-7.9231e+06,  0.0000e+00, -7.0000e+00],
        [ 0.0000e+00, -7.0000e+00,  0.0000e+00]], dtype=torch.int32)
    e: tensor([[ 0.,  0.,  0.],
        [ 0.,  0.,  0.]])
    """


# TODO ===========逐元素操作，此类操作输入输出一致，如绝对值等======================
"""常见：abs/sqrt/div/exp/fmod/log/pow/cos/sin/asin/atan2/cosh/ceil/round/floor/trunc"""
def function3():
    x = t.arange(0, 6, 1).view(2, 3)
    print(x)
    """
    输出：tensor([[ 0.,  1.,  2.],
        [ 3.,  4.,  5.]])
    """
    a = t.cos(x)
    print(a)
    """
    输出：tensor([[ 1.0000,  0.5403, -0.4161],
        [-0.9900, -0.6536,  0.2837]])
    """
    b = x % 3  # 等价于 t.fmod(x,3)
    print(b)
    """
    输出：tensor([[ 0.,  1.,  2.],
        [ 0.,  1.,  2.]])
    """
    c = x ** 2  # 等价于t.pow(x,2)
    print(c)
    """
    输出：tensor([[  0.,   1.,   4.],
        [  9.,  16.,  25.]])
    """
    d=t.clamp(c,min=2,max=15)#c中小于min的值为min,大于max 的值为max,小于max,大于min的值为x
    print(d)
    """
    tensor([[  2.,   2.,   4.],
        [  9.,  15.,  15.]])
    """

# TODO ===========归并操作，此类操作输入一般大于输出，如求和sum等======================
"""常见如：mean/sum/median/mode/norm/dist/std/var/cumsum/cumprod"""
def function4():
    x=t.ones(2,3)
    print(x)
    """
    输出：tensor([[ 1.,  1.,  1.],
        [ 1.,  1.,  1.]])
    """
    b=x.sum(dim=0,keepdim=True)#keepdim表示保留这个维度，2*3变成1*3
    print("b:",b)
    """
    b: tensor([[ 2.,  2.,  2.]])
    """
    c=x.sum(dim=0,keepdim=False)
    print("c:",c)
    """
    c: tensor([ 2.,  2.,  2.])
    """
    d=x.sum(dim=1)
    print("d:",d)#dim默认为False,2*3变成 2
    """
    d: tensor([ 3.,  3.])
    """
    y=t.arange(0,6).view(2,3)
    e=y.cumsum(dim=1)#沿行累加
    print("e:",e)
    """
    e: tensor([[  0.,   1.,   3.],
        [  3.,   7.,  12.]])
    """

# TODO ===========比较操作，如< ,>等======================
"""常见如：gt/ge/le/eq/ne/topk/sort/max/min"""
def function5():
    x=t.linspace(0,15,6).view(2,3)
    print(x)
    """
    tensor([[  0.,   3.,   6.],
        [  9.,  12.,  15.]])
    """
    y=t.linspace(15,0,6).view(2,3)
    print(y)
    a=x>y
    print("a:",a)
    """
    a: tensor([[ 0,  0,  0],
        [ 1,  1,  1]], dtype=torch.uint8)
    """
    b=x[x>y]
    print("b:",b)
    """ b: tensor([  9.,  12.,  15.]) """
    c=t.max(x)#返回指定tensor中最大的一个元素
    d = t.max(x, y)  # 比较返回两个tensor中较大的数
    e=t.max(x,dim=1)#返回指定维度上最大的值，返回tensor和下标
    print("c:",c)
    print("d:", d)
    print("e:",e)
    """c: tensor(15.)
        d: tensor([[ 15.,  12.,   9.],
        [  9.,  12.,  15.]])
        e: (tensor([  6.,  15.]), tensor([ 2,  2]))
    """
    f=t.clamp(x,min=10)
    print("f:",f)#比较x和10中较大的元素
    """
    f: tensor([[ 10.,  10.,  10.],
        [ 10.,  12.,  15.]])
    """

# TODO ===========线性代数======================
"""
trace 对角线元素之和
diag  对角线元素
triu/tril  矩阵的上三角/下三角，可指定偏移量
mm/bmm  矩阵乘法 、batch矩阵乘法
addmm/addbmm/addmv
t  转置
dot/cross  内积，外积
inverse 求逆
svd 奇异值分解
"""
def function6():
    x = t.linspace(0, 15, 6).view(2, 3)
    print(x)
    a=x.t()
    print("a:",a)
    print("是否连续存储空间：",a.is_contiguous())
    d=a.contiguous()
    print("d:",d)
    """
    a: tensor([[  0.,   9.],
        [  3.,  12.],
        [  6.,  15.]])
    是否连续存储空间： False
    d: tensor([[  0.,   9.],
        [  3.,  12.],
        [  6.,  15.]])
    """


#TODO  ==========================线性回归小栗子========================
def get_fake_data(batch_size=8):
    """产生随机数据，y=x*2+3+a ，a为随机噪声"""
    x=t.rand(batch_size,1)*20
    y=x*2+(1+t.randn(batch_size,1))*3#y=x*2+3+t.randn(batch_size,1)*3
    return x,y
def showNodes():
    x, y = get_fake_data()
    plt.scatter(x.squeeze().numpy(),y.squeeze().numpy())
    plt.show()#输出查看点
def demo():
    w=t.rand(1,1)
    b=t.zeros(1,1)
    lr=0.001#学习率
    for li in range(2000):
        x, y = get_fake_data()
        #forward 计算Loss
        y_pred=x.mm(w)+b.expand_as(y)
        loss=0.5*(y_pred-y)**2
        loss=loss.sum()
        #backward 手动计算梯度
        dloss=1
        dy_pred=dloss*(y_pred-y)

        dw=x.t().mm(dy_pred)
        db=dy_pred.sum()

        #更新参数
        w.sub_(lr*dw)
        b.sub_(lr*db)
        if li%100==0:
            x=t.arange(0,20).view(-1,1)
            y=x.mm(w)+b.expand_as(x)
            plt.ion()
            plt.plot(x.numpy(),y.numpy())#predict
            x2,y2=get_fake_data(batch_size=20)
            plt.scatter(x2.numpy(),y2.numpy())#true data
            plt.xlim(0,20)
            plt.ylim(0,41)
            plt.draw()#todo 和plt.ion()配合可以实现页面动态添加
            plt.pause(0.5)
            print(w.squeeze()[0],b.squeeze()[0])



if __name__ == "__main__":
    demo()
    # showNodes()
    #6
    # function6()
    #5
    # function5()
    #4
    # function4()
    # 3
    # function3()
    # 2
    # function2()
    # 1
    # function1()
