from torch import nn
from torch.nn import functional as F
from .BasicModule import BasicModule

"""之前有实现过，直接拿来用"""

class ResidualBlock(nn.Module):
    """
    实现子Module，Residual Block
    """

    def __init__(self, inchannel, outchannel, stride=1, shortcut=None):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(in_channels=inchannel, out_channels=outchannel, kernel_size=3, stride=stride, padding=1,
                      bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=outchannel, out_channels=outchannel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(outchannel)
        )
        self.right = shortcut

    def forward(self, x):
        out = self.left(x)
        residual = x if self.right is None else self.right(x)
        out += residual
        return F.relu(out)


class ResNet34(BasicModule):
    """
    实现主Module：resnet34
    resnet34包含多个layer，每个layer又包含多个residual block
    用子module实现residual block，用make_layer函数实现layer
    这里的layer更像一组网络层的组合
    """

    def __init__(self, num_class=1000):
        super(ResNet34, self).__init__()
        # 前几层图像转换
        self.pre = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        # 重复的layer，分别有3,4,6 共3个residual block
        self.layer1 = self.make_layer(64, 128, 3)
        self.layer2 = self.make_layer(128, 256, 4, stride=2)
        self.layer3 = self.make_layer(256, 512, 6, stride=2)
        self.layer4 = self.make_layer(512, 512, 3, stride=2)

        # 分类用的全连接
        self.fc = nn.Linear(512, num_class)

    def make_layer(self, inchannel, outchannel, block_num, stride=1):
        """
        构建layer ，包含多个residual block
        :param inchannel: 
        :param outchannel: 
        :param block_num: 
        :param stride: 
        :return: 
        """
        shortcut = nn.Sequential(
            nn.Conv2d(in_channels=inchannel, out_channels=outchannel, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(outchannel)
        )
        layers = []
        layers.append(ResidualBlock(inchannel, outchannel, stride, shortcut))
        for i in range(1, block_num):
            layers.append(ResidualBlock(outchannel, outchannel))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.pre(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = F.avg_pool2d(x, 7)
        x = x.view(x.size(0), -1)
        return self.fc(x)
