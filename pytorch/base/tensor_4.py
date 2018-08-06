
import torch as t

"""
主要讲Tensor的内容:
创建Tensor
Tensor 的常用操作,调整tensor的形状
索引操作
gather(input,dim,index)操作
scatter_操作，类似于gather操作的逆操作
"""



#TODO ===================创建Tensor===================

"""t.Tensor(*sizes)创建tensor时不会马上分配空间只会计算剩余内存是否足够使用到的时候才会分配，其他方式创建tensor是马上进行空间分配"""

#TODO Tensor(*size)创建Tensor
def function1():
    #指定tensor的形状
    a=t.Tensor(2,3)
    print(a)
    """
    输出：tensor([[ 5.2431e-27,  8.2116e-43,  4.7834e-33],
        [ 8.2116e-43,  4.7629e-33,  8.2116e-43]])
    """
#TODO list创建tensor
def function2():
    #使用list创建tensor
    list=[[1,2,3],[4,5,6]]
    b=t.Tensor(list)
    print(b)
    """
    输出：tensor([[ 1.,  2.,  3.],
        [ 4.,  5.,  6.]])
    """
    c=b.tolist()#将tensor转化为list
    print(c)
    """
    输出：[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    """
#TODO 查看tensor的Size
def function3():
    #tensor.size()会返回torch.Size对象，他是tuple的子类，与tuple略有区别
    b = t.Tensor([[1,2,3],[4,5,6]])
    b_size=b.size()
    b_shape=b.shape
    print(b_size)
    print(b_shape)
    """
    b.size()等价于b.shape
    输出：torch.Size([2, 3])
    """
    #b中元素的总个数，2*3，等价于b.nelement()
    n=b.numel()
    m=b.nelement()
    print("n:",n)#n: 6
    print("m:", m)#m: 6
    #创建一个跟b一样的tensor
    c=t.Tensor(b_size)
    #创建一个元素为2和3的tensor
    d=t.Tensor((2,3))
    e=t.Tensor([2,3])
    print("c:",c,"\n","d:",d,"\n","e:",e)
    """
    c: tensor([[ 3.2927e+10,  0.0000e+00,  3.2927e+10],
        [ 0.0000e+00,  3.2927e+10,  0.0000e+00]]) 
    d: tensor([ 2.,  3.]) 
    e: tensor([ 2.,  3.])
    """

#TODO 其他方式创建Tensor
def function4():
    a=t.ones(2,3)
    print(a)
    """
    输出：tensor([[ 1.,  1.,  1.],
        [ 1.,  1.,  1.]])
    """
    b=t.zeros(2,3)
    print(b)
    """
    输出：tensor([[ 0.,  0.,  0.],
        [ 0.,  0.,  0.]])
    """
    c=t.arange(1,6,1)#1-6每1个数，含头不含尾
    print(c)
    """
    输出：tensor([ 1.,  2.,  3.,  4.,  5.])
    """
    d=t.linspace(1,10,3)#1-10三等分点，含头尾
    print(d)
    """
    输出：tensor([  1.0000,   5.5000,  10.0000])
    """
    e=t.randn(2,3)#randn是生成均值为0，方差为1的正态分布
    print(e)
    """
    输出：tensor([[ 0.4382, -0.4589,  1.1424],
        [ 0.2180,  1.8085,  0.5358]])
    """
    f=t.rand(2,3)#rand是0-1均匀分布
    print(f)
    """
    输出：tensor([[ 0.6747,  0.7352,  0.3102],
        [ 0.2428,  0.2443,  0.6039]])
    """
    g=t.randperm(5)#返回一个从 0 to n - 1 的整数的随机排列.
    print(g)
    """
    输出：tensor([ 1,  3,  4,  2,  0])
    """
    h=t.eye(2,3)#对角线为1，不要求行列一致
    print(h)
    """
    tensor([[ 1.,  0.,  0.],
        [ 0.,  1.,  0.]])
    """


#TODO =========Tensor 的常用操作====================

#tensor.view方法可以调整tensor的形状，但是调整前后的元素总数必须一致，view不会修改自身数据，返回的tensor与原tensor 共享内存
#unsqueeze增加维度，squeeze减少维度
def function5():
    x=t.arange(0,6)
    a=x.view(2,3)
    print(a)
    """
    输出：tensor([[ 0.,  1.,  2.],
        [ 3.,  4.,  5.]])
    """
    b=a.view(-1,3)#当某一维度为-1时，会自动计算大小
    print(b)
    """
    输出：tensor([[ 0.,  1.,  2.],
        [ 3.,  4.,  5.]])
    """
    c=b.unsqueeze(1)#增加一个维度，下标从0开始，这个表示从第‘1’维度上加一个维度，即变成2*1*3
    print(c)
    """
    输出：tensor([[[ 0.,  1.,  2.]],

        [[ 3.,  4.,  5.]]])
    """
    d=b.unsqueeze(-2)#表示从倒数第二个维度增加一个维度
    print(d)
    """
    输出：tensor([[[ 0.,  1.,  2.]],

        [[ 3.,  4.,  5.]]])
    """
    e=b.view(1,1,1,2,3)
    print(e)
    """
    输出：tensor([[[[[ 0.,  1.,  2.],
           [ 3.,  4.,  5.]]]]])
    """
    f=e.squeeze(0)#压缩第0个维度
    print(f)
    """
    输出：tensor([[[[ 0.,  1.,  2.],
          [ 3.,  4.,  5.]]]])
    """
    g=e.squeeze()#压缩所有维度为1的维度，1*1*1*2*3这个1
    print(g)
    """
    输出：tensor([[ 0.,  1.,  2.],
        [ 3.,  4.,  5.]])
    """
    a[1][2]=100
    print(b)#a,b共享内存，a变了，b也变了
    """
    输出：tensor([[   0.,    1.,    2.],
        [   3.,    4.,  100.]])
    """

#TODO 通过resize修改size，这个不受尺寸的限制，多不显示少补
def function6():
    x=t.arange(0,6)
    a=x.view(2,3)
    a.resize_(1,3)#注意带_表示修改自己
    print(a)
    """
    输出：tensor([[ 0.,  1.,  2.]])
    """
    a.resize_(3,3)
    print(a)
    """说明上面的操作并没有把数据弄丢，对于多出的数据会分配新空间
    输出：tensor([[ 0.,  1.,  2.],
        [ 3.,  4.,  5.],
        [ 0.,  0.,  0.]])
    """

#TODO =================索引操作========================

def function7():
    x=t.arange(0,6,1)
    x=x.view(2,3)
    print(x)
    """
    输出：tensor([[ 0.,  1.,  2.],
        [ 3.,  4.,  5.]])
    """
    print(x[0])
    """输出第0行：tensor([ 0.,  1.,  2.])"""
    print(x[:,0])
    """输出第0列：tensor([ 0.,  3.])"""
    print(x[0][0])
    print(x[0,0])
    """输出第0行第0列：tensor(0.)"""
    print(x[0,-1])
    """输出第0行最后一个元素：tensor(2.)"""
    print(x[:2])
    """输出前两行：
    tensor([[ 0.,  1.,  2.],
        [ 3.,  4.,  5.]])
    """
    a=x>3
    print(a)
    """输出：
    tensor([[ 0,  0,  0],
        [ 0,  1,  1]], dtype=torch.uint8)
    """
    #等价于x.masked_select(x>3)
    b=x[x>3]
    c=x.masked_select(x>3)
    print(b)
    print(c)
    """输出：
    tensor([ 4.,  5.])
    """
    d=t.nonzero(x)#非0元素的下标
    print(d)
    """输出：
    tensor([[ 0,  1],
        [ 0,  2],
        [ 1,  0],
        [ 1,  1],
        [ 1,  2]])
    """

#TODO gather(input,dim,index)操作，根据index，在dim维度上选取数据，输出的size与index一致
def function8():
    a=t.arange(0,16).view(4,4)
    print(a)
    """
    输出：tensor([[  0.,   1.,   2.,   3.],
                [  4.,   5.,   6.,   7.],
                [  8.,   9.,  10.,  11.],
                [ 12.,  13.,  14.,  15.]])
    """
    index =t.LongTensor([[0,1,2,3]]).t()#转置
    print(index)
    """
    输出：tensor([[ 0],
        [ 1],
        [ 2],
        [ 3]])
    """
    b=a.gather(1,index)#获取对角线元素
    print(b)
    """输出：
    tensor([[  0.],
        [  5.],
        [ 10.],
        [ 15.]])
    """
    index=t.LongTensor([[3,2,1,0]]).t()
    print(index)
    """
    输出：tensor([[ 3],
        [ 2],
        [ 1],
        [ 0]])
    """
    c=a.gather(1,index)#获取反对角线上的数
    print(c)
    """
    输出：tensor([[  3.],
        [  6.],
        [  9.],
        [ 12.]])
    """
    index=t.LongTensor([[3,2,1,0]])#注意这里没有转置
    print(index)
    """输出：tensor([[ 3,  2,  1,  0]])"""
    d = a.gather(0, index)  # 获取反对角线上的数
    print(d)
    """输出：tensor([[ 12.,   9.,   6.,   3.]])"""
    index=t.LongTensor([[0,1,2,3],[3,2,1,0]]).t()#获取两个对角线上的元素
    f=a.gather(1,index)
    print(f)
    """
    输出：tensor([[  0.,   3.],
        [  5.,   6.],
        [ 10.,   9.],
        [ 15.,  12.]])
    """

#TODO scatter_操作，类似于gather操作的逆操作,注意带下划线，是修改本数据的操作
def function9():
    a = t.arange(0, 16).view(4, 4)
    index = t.LongTensor([[0, 1, 2, 3], [3, 2, 1, 0]]).t()  # 获取两个对角线上的元素
    f = a.gather(1, index)
    c=t.zeros(4,4)
    c.scatter_(1,index,f)
    d=c.scatter(1,index,f)
    print(c)
    print(d)
    """
    输出：
    tensor([[  0.,   0.,   0.,   3.],
        [  0.,   5.,   6.,   0.],
        [  0.,   9.,  10.,   0.],
        [ 12.,   0.,   0.,  15.]])
    """

if __name__=="__main__":
    #9
    function9()
    #8
    # function8()
    #7
    # function7()
    #6
    # function6()
    #5
    # function5()
    # 4
    # function4()
    # 3
    # function3()
    #2
    # function2()
    #1
    # function1()



