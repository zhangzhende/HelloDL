import visdom
import time
import numpy as np


class Visualizer(object):
    """
    封装了visdom 的基本操作，但是仍然可以通过“self.vis.function”或者"self.function"调用原生的visdom接口
    """
    def __init__(self,env="default",**kwargs):
        self.vis=visdom.Visdom(env=env,**kwargs)#这一步操作使其如注释所示
        #保存("loss",23) 即loss的第23个节点
        self.index={}
        self.log_text=""

    def reinit(self,env="default",**kwargs):
        """
        修改visdom的配置，重新初始化
        :param env: 
        :param kwargs: 
        :return: 
        """
        self.vis=visdom.Visdom(env=env,**kwargs)
        return self

    def plot_many(self,d):
        """
        一次plot多个
        :param d: dict(name,value)  i.e. ("loss",0.11)
        :return: 
        """
        for k,v in d.item():
            self.plot(k,v)

    def plot(self,name,y,**kwargs):
        """
        self.plot("loss",0.11)
        :param name: 
        :param kwargs: 
        :return: 
        """
        x=self.index.get(name,0)
        self.vis.line(Y=np.array([y]),X=np.array([x]),win=(name),opts=dict(title=name),update=None if x ==0 else "append",**kwargs)
        self.index[name]=x+1

    def img_many(self,d):
        for k,v in d.items():
            self.img(k,v)

    def img(self,name,img_,**kwargs):
        """
        可以这么传参数：
        self.img('input_img',t.Tensor(64,64))
        self.img('input_imgs',t.Tensor(3,64,64))
        self.img('input_imgs',t.Tensor(100,1,64,64))
        self.img('input_imgs',t.Tensor(100,3,64,64),nrows=10)
        不要这样：self.img('input_imgs',t.Tensor(100,64,64),nrows=10)
        :param name: 
        :param img_: 
        :param kwargs: 
        :return: 
        """
        self.vis.images(img_.cpu().numpy(),
                        win=(name),
                        opts=dict(title=name),**kwargs)

    def log(self,info,win="log_text"):
        """
        self.log({"loss":1,"lr":0.0001})
        :param info: 
        :param win: 
        :return: 
        """
        self.log_text+=("[{time}]{info}<br>".format(time=time.strftime("%m%d_%H%M%S"),
                                                    info=info))
        self.vis.text(self.log_text,win=win)

    def __getattr__(self, name):
        """
        自定义的plot,image,log,plot_many等除外
        self.function 等价于self.vis.function
        :param item: 
        :return: 
        """
        return getattr(self.vis,name=name)






