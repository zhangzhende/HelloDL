import os
from PIL import Image
from torch.utils import data
import numpy as np
from torchvision import transforms as T

"""数据加载这里的图片是猫狗在一起，按照文件名带有猫狗来区分类型"""

class DogCat(data.Dataset):
    def __init__(self,root,transforms=None,train=True,test=False):
        """
        目标：获取所有图片地址，并根据训练、验证、测试划分数据
        """
        self.test=test
        imgs=[os.path.join(root,img) for img in os.listdir(root)]
        #路径说明：test:data/test1/111.jpg
        #        train:data/train/cat.1111.jpg  注意不同在下面处理截取的时候处理方式不一样
        if self.test:
            imgs=sorted(imgs,key=lambda x: int(x.split(".")[-2].split("/")[-1]))#拿到序号，按照序号排列图片
        else:
            imgs=sorted(imgs,key=lambda x:int(x.split(".")[-2]))#拿到序号,按照序号排列图片
        imgs_num=len(imgs)

        #划分训练，验证，测试数据集，训练：验证=7:3
        if self.test:
            self.imgs=imgs
        elif train:
            self.imgs=imgs[:int(0.7*imgs_num)]
        else:
            self.imgs=imgs[int(0.7*imgs_num):]

        if transforms is None:
            #数据转换操作，但是测试验证和训练的数据转换不同
            normalize=T.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])

        #对于测试集和验证集
            if self.test or train:
                self.transforms=T.Compose([
                    T.Resize(224),
                    T.CenterCrop(224),
                    T.ToTensor(),
                    normalize
                ])
            #训练集
            else:
                self.transforms=T.Compose([
                    T.Resize(256),
                    # T.RandomSizedCrop(224), #deprecated
                    T.RandomResizedCrop(224),
                    T.RandomHorizontalFlip(),
                    T.ToTensor(),
                    normalize
                ])
    def __getitem__(self, index):
        """
        返回一张图片的数据
        如果是测试集，没有图片ID，如1000.jpg,返回1000
        :param index: 
        :return: 
        """
        img_path=self.imgs[index]
        if self.test:
            label=int(self.imgs[index].split(".")[-2].split("/")[-1])
        else:
            label=1 if "dog" in img_path.split("/")[-1] else 0
        data=Image.open(img_path)
        data=self.transforms(data)
        return data,label

    def __len__(self):
        """
        返回数据集中所有图片的个数
        :return: 
        """
        return len(self.imgs)