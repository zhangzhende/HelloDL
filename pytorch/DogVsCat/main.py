import os
import torch as t
# from .config import opt
# from .utils.visualize import Visualizer
# from .data.dataset import DogCat
import models
from torch.utils.data import DataLoader
from torchnet import meter
from torch.autograd import Variable
from torch.nn import functional as F
from  config import opt
from data.dataset import DogCat
from utils.visualize import Visualizer
"""
训练主要步骤：
定义网络
定义数据
定义损失函数和优化器
计算重要指标
开始训练 --训练网络
        --可视化各种指标
        --计算在验证集上的指标

"""
#TODO 训练
def train(**kwargs):
    #根据命令行参数更新配置
    opt.parse(kwargs=kwargs)
    vis=Visualizer(opt.env)

    #1.模型
    model=getattr(models,opt.model)()#注意实例化
    if opt.load_model_path:
        model.load(opt.load_model_path)
    if opt.use_gpu:
        model.cuda()

    #2 数据
    train_data=DogCat(opt.train_data_root,train=True)
    val_data=DogCat(opt.train_data_root,train=False)
    train_dataloader=DataLoader(dataset=train_data,
                                batch_size=opt.batch_size,
                                shuffle=True,
                                num_workers=opt.num_workers)
    val_dataloader = DataLoader(dataset=val_data,
                                  batch_size=opt.batch_size,
                                  shuffle=False,
                                  num_workers=opt.num_workers)

    #3 损失函数和优化器
    criterion=t.nn.CrossEntropyLoss()
    lr=opt.lr
    optimizer=t.optim.Adam(params=model.parameters(),
                           lr=lr,
                           weight_decay=opt.weight_decay)

    #4 统计指标：平滑处理之后的损失，还有混淆矩阵
    loss_meter=meter.AverageValueMeter()#AverageValueMeter,计算所有数的平均值
    confusion_matrix=meter.ConfusionMeter(2)#用来统计二分类的一些统计指标
    previous_loss=1e100

    #训练
    for epoch in range(opt.max_epoch):
        loss_meter.reset()
        confusion_matrix.reset()
        for ii ,(data,label) in enumerate(train_dataloader):
            #训练模型参数
            input=Variable(data)
            target=Variable(label)
            if opt.use_gpu:
                input=input.cuda()
                target=target.cuda()
            optimizer.zero_grad()
            score=model(input)
            loss=criterion(score,target)
            loss.backward()
            optimizer.step()

            #更新统计指标及可视化
            loss_meter.add(loss.item())
            confusion_matrix.add(predicted=score.data,target=target)

            if ii%opt.print_frep==opt.print_frep-1:
                vis.plot("loss",loss_meter.value()[0])

                #如果需要的话进入debug模式
                if os.path.exists(opt.debug_file):
                    import ipdb
                    ipdb.set_trace()
        model.save()
        #计算验证集上的指标以及可视化
        val_cm,val_accuracy=val(model,val_dataloader)
        vis.plot("val_accuracy",val_accuracy)
        vis.log("epoch:{epoch},lr:{lr},loss:{loss},train_cm:{train_cm},val_cm:{val_cm}".format(
            epoch=epoch,
            loss=loss_meter.value()[0],
            val_cm=str(val_cm.value()),
            train_cm=str(confusion_matrix.value()),
            lr=lr
        ))
        #如果损失不再下降，则降低学习率
        if loss_meter.value()[0] > previous_loss:
            lr=lr*opt.lr_decay
            for param_group in optimizer.param_groups:
                param_group["lr"]=lr
        previous_loss=loss_meter.value()[0]


#TODO 计算模型在验证集上的准确率等信息，用以辅助训练
def val(model,dataloader):
    """
    计算模型在验证集上的准确率等信息
    验证：注意需要将模型设置与验证模式model.eval，验证完毕后需要将模式改回训练模式model.train
    :return: 
    """
    #吧模型设置为验证模式
    model.eval()

    confusion_matrix=meter.ConfusionMeter(2)
    for ii,data in enumerate(dataloader):
        input ,label=data
        val_input=Variable(input,volatile=True)
        val_label=Variable(label.long(),volatile=True)
        if opt.use_gpu:
            val_input=val_input.cuda()
            val_label=val_label.cuda()
        score =model(val_input)
        confusion_matrix.add(score.data.squeeze(),label.long())
    #将复制模式改为训练模式
    model.train()

    cm_value=confusion_matrix.value()
    accuracy=100*(cm_value[0][0]+cm_value[1][1])/cm_value.sum()
    return confusion_matrix,accuracy

#TODO 测试
def test(**kwargs):
    """
    计算每个样本属于狗的概率，并保存为CSV文件
    :return: 
    """
    opt.parse(kwargs)
    #模型
    model =getattr(models,opt.model)().eval()
    if opt.load_model_path:
        model.load(opt.load_model_path)
    if opt.use_gpu:
        model.cuda()
    #数据
    train_data=DogCat(opt.test_data_root,test=True)
    test_dataloader=DataLoader(
        dataset=train_data,
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.num_workers
    )
    result=[]
    for II,(data,path) in enumerate(test_dataloader):
        input=Variable(data,volatile=True)
        if opt.use_gpu:
            input=input.cuda()
        score=model(input)
        probability=F.softmax(score)[:,1].data.tolist()
        batch_results=[(path_,probability_) for path_,probability_ in zip(path,probability)]
        result+=batch_results
        write_csv(result,opt.result_file)
    return result

def write_csv(result,file_name):
    import csv
    with open(file_name,"w") as f:
        writer=csv.writer(f)
        writer.writerow(["id","label"])
        writer.writerows(result)

#TODO 打印帮助提示信息
def help():
    """
    打印帮助额信息：python file.py help
    :return: 
    """
    print("""
    usage:python {0} <function> [--args=value,]
    <function>:= train|test|help
    example:
        python {0} trian --env='env0701' --lr=0.01
        python {0} test ==dataser='path/to/dataset/root/'
        python {0} help
    avaiable args:
    """.format(__file__))
    from inspect import getsource
    source=(getsource(opt.__class__))
    print(source)

if __name__=="__main__":
    import fire
    fire.Fire()
    train()