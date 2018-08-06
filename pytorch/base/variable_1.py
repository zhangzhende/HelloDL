import torch as t
from torch.autograd import Variable as V

"""
0.4版本之后Tensor和Variable合并
在此之前只有Variable支持backward，Variable是Tensor的封装。
新版本中，torch.autograd.Variable和torch.Tensor将同属一类。更确切地说，
torch.Tensor 能够跟踪历史并像旧版本的 Variable 那样运行; Variable 封装仍旧可以像以前一样工作，
但返回的对象类型是 torch.Tensor。 这意味着你不再需要代码中的所有变量封装器。

"""
#TODO Variable的一些属性
def function1():
    a=V(t.ones(3,4),requires_grad=True)
    print("a:",a)
    """a: tensor([[ 1.,  1.,  1.,  1.],
        [ 1.,  1.,  1.,  1.],
        [ 1.,  1.,  1.,  1.]])
    """
    b=V(t.zeros(3,4))
    print("b:",b)
    """
    b: tensor([[ 0.,  0.,  0.,  0.],
        [ 0.,  0.,  0.,  0.],
        [ 0.,  0.,  0.,  0.]])
    """
    c=a.add(b)
    print("c:",c)
    """
    c: tensor([[ 1.,  1.,  1.,  1.],
        [ 1.,  1.,  1.,  1.],
        [ 1.,  1.,  1.,  1.]])
    """
    d=c.sum()
    d.backward()
    print("d:",d)
    """d: tensor(12.)"""
    e=c.data.sum()#取data后成为tensor
    f=c.sum()#任然还是个Variable
    print("e:",e)
    print("f:",f)
    g=a.grad
    print("g:",g)
    """g: tensor([[ 1.,  1.,  1.,  1.],
        [ 1.,  1.,  1.,  1.],
        [ 1.,  1.,  1.,  1.]])
    """
    #Variable创建时默认是不需要求导的，指定a求导了requires_grad属性为true，c依赖于a，所以c也需要求导，b创建时没有指定，所以默认为false
    print(a.requires_grad,b.requires_grad,c.requires_grad)#True False True
    #是否叶子节点，由用户创建的Variable是叶子节点，对应的grad_fn是None,由基础variable组合计算而成的不是
    print(a.is_leaf,b.is_leaf,c.is_leaf)#True True False

def function2():
    def f(x):
        y=x**2 * t.exp(x)
        return y
    def gradf(x):
        dx=2*x*t.exp(x)+x**2*t.exp(x)
        return dx
    x=V(t.randn(3,4),requires_grad=True)
    y=f(x)
    print("y:",y)
    y.backward(t.ones(y.size()))#grad_variables形状一致
    a=x.grad#自动的求导
    b=gradf(x)#手动的求导
    print("a:",a)
    print("b:",b)
    """
    y: tensor([[ 1.3254,  2.8313,  0.0282,  0.4961],
        [ 2.8951,  0.2237,  0.1450,  0.4676],
        [ 0.0749,  0.3396,  0.3034,  0.0730]])
    a: tensor([[ 4.7256,  8.4177,  0.3915, -0.1807],
        [ 8.5654, -0.4574, -0.4525, -0.2361],
        [-0.3913, -0.3941, -0.4215, -0.3883]])
    b: tensor([[ 4.7256,  8.4177,  0.3915, -0.1807],
        [ 8.5654, -0.4574, -0.4525, -0.2361],
        [-0.3913, -0.3941, -0.4215, -0.3883]])
    """

def function3():
    x=V(t.ones(1))
    w=V(t.rand(1),requires_grad=True)
    y=x*w
    #x默认创建，属性为false，w 指定为true，y依赖于w,所以为true
    print(x.requires_grad, w.requires_grad, y.requires_grad)#False True True
def function4():
    x=V(t.ones(1),volatile=True)#volatile was removed and now has no effect. Use `with torch.no_grad():` instead.
    w=V(t.rand(1),requires_grad=True)
    y=x*w
    #x默认创建，属性为false，w 指定为true，y依赖于w,所以为true
    print(x.requires_grad, w.requires_grad, y.requires_grad)#False True True

if __name__ =="__main__":
    #4
    function4()
    #3
    # function3()
    #2
    # function2()
    #1
    # function1()