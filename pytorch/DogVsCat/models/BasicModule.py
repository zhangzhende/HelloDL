import torch as t
import time

"""basicmodule是对nn.Module 的简单封装，提供快速加载和保存模型的接口"""


class BasicModule(t.nn.Module):
    def __init__(self):
        super(BasicModule, self).__init__()
        self.model_name = str(type(self))

    def load(self,path):
        """
        可以加载指定路径的模型
        :param path: 
        :return: 
        """
        self.load_state_dict(t.load(path))

    def save(self, name=None):
        """
        保存模型，默认使用“模型名字+时间”作为文件名，如AlexNet_0710_23:57:29.pth
        :param name: 
        :return: 
        """
        if name is None:
            prefix = "D:/workingspace/github/HelloDL/pytorch/DogVsCat/checkpoints/" + self.model_name + "_"
            name = time.strftime(prefix + "%m%d_%H_%M_%S.pth")
        t.save(self.state_dict(), name)
        return name
