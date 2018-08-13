import torch as t
import os
from PIL import Image
import numpy as np
from torch.utils import data
from torchvision import transforms as T
import matplotlib.pyplot as plt
import random



"""
pytorch 中的一些常用的工具

1.加载数据，如猫狗分类中将图片和标签对应上，图片为cat.12.jpg这样的格式，名字中包含cat就是猫，
"""


"""===================1.加载图片================================"""
#TODO 加载猫狗的图片
class DogCat(data.Dataset):
    def __init__(self,root):
        imgs=os.listdir(root)
        #所有图片的绝对路径
        #这里不实际的加载图片，只是指定路径，当调用__getitem__时才会真正读取图片
        self.imgs=[os.path.join(root,img) for img in imgs]

    def __getitem__(self, index):
        img_path=self.imgs[index]
        #如果文件名中包含dog则标签为1 ，否则标签为0
        label =1 if "dog" in img_path.split("/")[-1] else 0
        pil_img=Image.open(img_path)
        array=np.asarray(pil_img)
        data=t.from_numpy(array)
        return data,label
    def __len__(self):
        return len(self.imgs)

#TODO 测试一下猫狗文件加载
def function1():
    dataset=DogCat("../data/dogcat/")
    img,label=dataset[0]  # 相当于调用dataset.__getitem__(0)
    for img ,label in dataset:
        print(img.size(),img.float().mean(),label)


"""=========================1.归一化图片，加载图片=================================="""
#TODO 注意 方法过时 The use of the transforms.Scale transform is deprecated, please use transforms.Resize instead
#图片转换方式，缩放--》裁剪---》归一化——》归一化
transform=T.Compose([
    # T.Scale(224),   #缩放图片Image，保持长宽比不变，最短边为224像素
    T.Resize(224),
    T.CenterCrop(224),  #从图片中间切出224*224的图片
    T.ToTensor(),    #将图片image转换成Tensor，并归一化置[0,1]
    T.Normalize(mean=[0.5, 0.5, 0.5],std=[0.5, 0.5, 0.5]) #标准化[-1,1],并规定了均值和标准差
])
#TODO 上面的一种方式对于图片大小不一致的情况下无法使用，下面这个现将图片统一了
class DogCat2(data.Dataset):
    def __init__(self,root,transforms=None):
        imgs=os.listdir(root)
        #所有图片的绝对路径
        #这里不实际的加载图片，只是指定路径，当调用__getitem__时才会真正读取图片
        self.imgs=[os.path.join(root,img) for img in imgs]
        self.transforms=transforms

    def __getitem__(self, index):
        img_path=self.imgs[index]
        #如果文件名中包含dog则标签为1 ，否则标签为0
        label =1 if "dog" in img_path.split("/")[-1] else 0
        data=Image.open(img_path)
        if self.transforms:
            data=self.transforms(data)
        return data,label
    def __len__(self):
        return len(self.imgs)

def function2():
    dataset=DogCat2("../data/dogcat/",transforms=transform)
    image,label=dataset[0]
    for img ,label in dataset:
        print(image.size(),label)


"""=====================3.ImageFolder的使用  按文件夹分类，加载图片========================================="""

"""
ImageFolder( root, transform=None, target_transform=None, loader=default_loader)
root:指定路径寻找图片
transform: 对PIL Image进行转换的操作
target_transform：对label的转换
loader：指定加载图片的函数
"""
from torchvision.datasets import ImageFolder
dataset=ImageFolder("../data/dogcat2/")

def function3():
    print(dataset.class_to_idx)
    """{'cat': 0, 'dog': 1}"""
    print(dataset.imgs)
    """[('../data/dogcat2/cat\\cat.1.jpg', 0), ('../data/dogcat2/cat\\cat.11.jpg', 0), ('../data/dogcat2/cat\\cat.2.jpg', 0), ('../data/dogcat2/cat\\cat.3.jpg', 0), ('../data/dogcat2/cat\\cat.4.jpg', 0), ('../data/dogcat2/cat\\cat.8.jpg', 0), ('../data/dogcat2/cat\\cat.9.jpg', 0), ('../data/dogcat2/dog\\dog.14.jpg', 1), ('../data/dogcat2/dog\\dog.24.jpg', 1), ('../data/dogcat2/dog\\dog.31.jpg', 1), ('../data/dogcat2/dog\\dog.4.jpg', 1), ('../data/dogcat2/dog\\dog.73.jpg', 1), ('../data/dogcat2/dog\\dog.75.jpg', 1), ('../data/dogcat2/dog\\dog.80.jpg', 1)]
    """
def function4():
    #第一维是第几张图，第二维表示数据，第二维为1表示label，为0表示PIL Image
    label=dataset[0][1]
    print(label)
    image=dataset[0][0]
    image.show()

#TODO 方法过时 he use of the transforms.RandomSizedCrop transform is deprecated, please use transforms.RandomResizedCrop instead.
transform2=T.Compose([
    # T.RandomSizedCrop(224),  #先将给定的PIL.Image随机切，然后再resize成给定的size大小。
    T.RandomResizedCrop(224),
    T.RandomHorizontalFlip(),  #随机水平翻转给定的PIL.Image,概率为0.5。即：一半的概率翻转，一半的概率不翻转。
    T.ToTensor(),
    T.Normalize(mean=[0.4, 0.4, 0.4], std=[0.2, 0.2, 0.2])
])

dataset2=ImageFolder("../data/dogcat2/",transform=transform2)
def function5():
    print(dataset2[0][0].size())
    """torch.Size([3, 224, 224])"""
    toimg=T.ToPILImage()
    image=toimg(dataset2[0][0]*0.2+0.4)
    image.show()


"""==================4.DataLoader 来加入图片，包括batch，shuffle等操作===================================="""
"""
DataLoader( dataset, batch_size=1, shuffle=False, sampler=None, batch_sampler=None,
                 num_workers=0, collate_fn=default_collate, pin_memory=False, drop_last=False,
                 timeout=0, worker_init_fn=None)
"""
from torch.utils.data import DataLoader
dataloader=DataLoader(dataset=dataset2,batch_size=3,shuffle=True,num_workers=0,drop_last=False)

def function6():
    dataiter=iter(dataloader)
    imgs,labels=next(dataiter)
    print(imgs.size())  #batch_size,channel,height,weight
"""
dataloader 是一个可迭代的对象，可以像使用迭代器一样使用他
方法1：
    dataiter=iter(dataloader)
    imgs,labels=next(dataiter)
方法2：
    for batch_datas,batch_labels in dataloader:
    train()
"""

class NewDogCat(DogCat2):
    def __getitem__(self, index):
        try:
            #调用父类的获取函数
            return super(NewDogCat, self).__getitem__(index=index)
        except:
            #如果发生异常就返回None，None
            return None,None

from torch.utils.data.dataloader import default_collate
def my_default_collate(batch):
    """
    batch 中每个元素都是形如（data,label）
    :param batch: 
    :return: 
    """
    #过滤为None的数据,filter(function, iterable)  function -- 判断函数。iterable -- 可迭代对象。
    batch=list(filter(lambda x:x[0] is not None,batch))
    return default_collate(batch)# 用默认的方式拼接过滤后的数据

dataset3=NewDogCat("../data/dogcat/",transforms=transform2)
def function7():
    print(dataset3[5])
    """(None, None)"""

def function8():
    dataloader2=DataLoader(dataset=dataset3,batch_size=2,collate_fn=my_default_collate,num_workers=1)
    for batch_datas,batch_labels in dataloader2:
        print(batch_datas.size(),batch_labels.size())
        """torch.Size([2, 3, 224, 224]) torch.Size([2])
            torch.Size([2, 3, 224, 224]) torch.Size([2])...
"""

#对于异常图片的另一种处理方式：如果有图片损坏，随机取一张图片替换
class NewDogCat2(DogCat2):
    def __getitem__(self, index):
        try:
            #调用父类的获取函数
            return super(NewDogCat, self).__getitem__(index=index)
        except:
            #如果发生异常就随机取一张照片替换
            new_index=random.randint(0,len(self)-1)
            return self[new_index]

"""===================5.随机取样，Sampler================================================"""
from torch.utils.data.sampler import WeightedRandomSampler
def function9():
    dataset4=DogCat2("../data/dogcat/",transforms=transform2)
    #狗的照片呗取出的概率是猫的2倍，两类图片被取出的概率与weights的绝对值无关，只和比值大小相关
    weights=[2 if label==1 else 1 for data ,label in dataset4]
    print(weights)
    """[1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2]"""
    sampler=WeightedRandomSampler(weights,num_samples=9,replacement=True)
    dataloader3=DataLoader(dataset=dataset4,batch_size=3,sampler=sampler)
    for datas,labels in dataloader3:
        print(labels.tolist())
    """replacement=True表示有样本是被重复返回的。
    [1, 0, 0]
    [1, 0, 1]
    [1, 1, 1]
    """
    sampler2=WeightedRandomSampler(weights=weights,num_samples=8,replacement=False)
    dataloader4=DataLoader(dataset=dataset4,batch_size=4,sampler=sampler2)
    for datas,labels in dataloader4:
        print(labels.tolist())
        """
        [1, 0, 0, 1]
        [0, 1, 1, 1]

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
    #4
    # function4()
    #3
    # function3()
    #2
    # function2()
    #1
    # function1()