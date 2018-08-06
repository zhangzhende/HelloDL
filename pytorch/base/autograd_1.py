import torch as t
from torch.autograd import Function
from torch.autograd import Variable as V
from matplotlib import pyplot as plt

""""
1.自定义反向求导
2.高阶求导
3.线性回归小栗子
"""



"""
绝大多数函数都可以使用autograd实现反向求导，，但是如果需要自己写一个复杂的函数，不支持自动反向求导的话
就要自己写的function 实现他的前向传播和反向传播
"""
#TODO 自定义反向求导
class MultiplyAdd(Function):
    # @staticmethod
    def forward(self, w,x,b):
        print("type in forward:",type(x))
        self.save_for_backward(w,x)
        output=w*x+b
        return output

    # @staticmethod
    def backward(self, grad_output):
        w,x=self.saved_tensors
        print("type in backward:",type(x))
        grad_w=grad_output*x
        grad_x=grad_output*w
        grad_b=grad_output*1
        return grad_w,grad_x,grad_b

#TODO 自定义反向求导使用
def function1():
    x=V(t.ones(1))
    w=V(t.rand(1),requires_grad=True)
    b=V(t.rand(1),requires_grad=True)
    print("开始前向传播")
    muadd=MultiplyAdd()#先实例化才行

    z=muadd(x,w,b)
    print("开始反向传播")
    z.backward()

#TODO 高阶求导
def function2():
    x=V(t.Tensor([5]),requires_grad=True)
    y=x**2
    #一阶导数
    grad_x=t.autograd.grad(y,x,create_graph=True)
    print(grad_x)
    #二阶导数
    grad_grad_x=t.autograd.grad(grad_x[0],x)
    print(grad_grad_x)

#TODO 用Variable实现线性回归
def get_fake_data(batch_size=8):
    x=t.rand(batch_size,1)*20
    y=x*2 +3+3*t.randn(batch_size,1)
    return x,y

def show_use():
    w=V(t.rand(1,1),requires_grad=True)
    b=V(t.zeros(1,1),requires_grad=True)
    lr=0.001#学习率

    for ii in range(800):
        x,y=get_fake_data()
        x,y=V(x),V(y)

        #forward 计算loss
        y_pred=x.mm(w)+b.expand_as(y)
        loss=0.5*(y_pred-y)**2
        loss=loss.sum()

        #backward 计算梯度
        loss.backward()

        #更新参数
        w.data.sub_(lr*w.grad.data)
        b.data.sub_(lr*b.grad.data)

        #梯度清零
        w.grad.data.zero_()
        b.grad.data.zero_()

        if ii%100==0:
            #画图
            x=t.arange(0,20).view(-1,1)
            y=x.mm(w.data)+b.data.expand_as(x)
            plt.ion()
            plt.plot(x.numpy(),y.numpy())#predicted

            x2,y2=get_fake_data(batch_size=20)
            plt.scatter(x2.numpy(),y2.numpy())#true data

            plt.xlim(0,20)
            plt.ylim(0,41)
            plt.draw()
            plt.pause(0.5)
            plt.figure().clear()
    print(w.data.item(),b.data.item())#w.data.squeeze()[0]这种表达方式会在0.5版本移除，换成item()


if __name__=="__main__":
    #线性回归
    show_use()
    #2
    # function2()
    #1
    # function1()