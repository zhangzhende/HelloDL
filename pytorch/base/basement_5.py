import torch as t


# TODO ===========tensor数据类型相关======================

# TODO 不好用了
def function1():
    t.set_default_tensor_type('torch.IntTensor')  # 修改默认的tensor类型
    #    only floating-point types are supported as the default type
    a = t.Tensor(2, 3)
    print(a)


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
    pass


if __name__ == "__main__":
    # 3
    function3()
    # 2
    # function2()
    # 1
    # function1()
