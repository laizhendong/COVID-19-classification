import os
import torch
import warnings
import numpy as np
import torch.nn as nn
from visdom import Visdom
import torch.optim as optim
import torchxrayvision as xrv
from torchvision import transforms
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from torch.optim.lr_scheduler import StepLR


# 在自定义的文件中导入方法

from tools.conduct import val
from tools.conduct import test
from tools.conduct import train
from tools.dataload import CovidCTDataset

#  预处理，标准化与图像增强

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 依通道标准化

train_transformer = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomResizedCrop((224), scale=(0.5, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize
])

val_transformer = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize
])

if __name__ == '__main__':

    batchsize = 24  # 原来用的10，这里改成24，根据GPU容量确定。
    total_epoch = 2000  # 2000个epoch，每个epoch 18次迭代，425/24，训练完就要迭代2000*18次
    votenum = 10

    # ------------------------------- step 1/5 数据 ----------------------------

    # 实例化CovidCTDataset
    trainset = CovidCTDataset(root_dir='C:/Users/LZD/Desktop/work/COVID-CT-master/Data',
                              txt_COVID='C:/Users/LZD/Desktop/work/COVID-CT-master/Data/trainCT_COVID.txt',
                              txt_NonCOVID='C:/Users/LZD/Desktop/work/COVID-CT-master/Data/trainCT_NonCOVID.txt',
                              transform=train_transformer)
    valset = CovidCTDataset(root_dir='C:/Users/LZD/Desktop/work/COVID-CT-master/Data',
                            txt_COVID='C:/Users/LZD/Desktop/work/COVID-CT-master/Data/valCT_COVID.txt',
                            txt_NonCOVID='C:/Users/LZD/Desktop/work/COVID-CT-master/Data/valCT_NonCOVID.txt',
                            transform=val_transformer)
    testset = CovidCTDataset(root_dir='C:/Users/LZD/Desktop/work/COVID-CT-master/Data',
                             txt_COVID='C:/Users/LZD/Desktop/work/COVID-CT-master/Data/testCT_COVID.txt',
                             txt_NonCOVID='C:/Users/LZD/Desktop/work/COVID-CT-master/Data/testCT_NonCOVID.txt',
                             transform=val_transformer)
    print(trainset.__len__())
    print(valset.__len__())
    print(testset.__len__())

    # 构建DataLoader
    train_loader = DataLoader(trainset, batch_size=batchsize, drop_last=False, shuffle=True)
    val_loader = DataLoader(valset, batch_size=batchsize, drop_last=False, shuffle=False)
    test_loader = DataLoader(testset, batch_size=batchsize, drop_last=False, shuffle=False)

    # ------------------------------ step 2/5 模型 --------------------------------

    model = xrv.models.DenseNet(num_classes=2, in_channels=3).cuda()  # DenseNet 模型，二分类
    modelname = 'DenseNet_medical'
    torch.cuda.empty_cache()

    # ----------------------------- step 3/5 损失函数 ----------------------------

    criteria = nn.CrossEntropyLoss()  # 二分类用交叉熵损失

    # ----------------------------- step 4/5 优化器 -----------------------------

    optimizer = optim.Adam(model.parameters(), lr=0.0001)  # 采用Adam优化器
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)  # 动态调整学习率策略，初始学习率0.0001

    # ----------------------------  step 5/5 训练 ------------------------------
    viz = Visdom(server='http://localhost/', port=8097)

    viz.line([[0., 0., 0., 0., 0.]], [0], win='train_performance', update='replace', opts=dict(title='train_performance', legend=['precision', 'recall', 'AUC', 'F1', 'acc']))
    viz.line([[0., 0.]], [0], win='train_Loss', update='replace', opts=dict(title='train_Loss', legend=['train_loss', 'val_loss']))

    warnings.filterwarnings('ignore')

    TP = 0
    TN = 0
    FN = 0
    FP = 0

    r_list = []
    p_list = []
    acc_list = []
    AUC_list = []

    vote_pred = np.zeros(valset.__len__())
    vote_score = np.zeros(valset.__len__())

    # 迭代3000*14次

    for epoch in range(1, total_epoch + 1):

        train_loss = train(optimizer, epoch, model, train_loader, modelname, criteria)  # 进行一个epoch训练的函数

        targetlist, scorelist, predlist, val_loss = val(model, val_loader, criteria)  # 用验证集验证
        print('target', targetlist)
        print('score', scorelist)
        print('predict', predlist)
        vote_pred = vote_pred + predlist
        vote_score = vote_score + scorelist
        if epoch % votenum == 0:  # 每10个epoch，计算一次准确率和召回率等

            # major vote
            vote_pred[vote_pred <= (votenum / 2)] = 0   # 投票，对某样本的预测，超过一半是正例，则判为正例，反之判为负例
            vote_pred[vote_pred > (votenum / 2)] = 1
            vote_score = vote_score / votenum

            print('vote_pred', vote_pred)
            print('targetlist', targetlist)


            TP = ((vote_pred == 1) & (targetlist == 1)).sum()
            TN = ((vote_pred == 0) & (targetlist == 0)).sum()
            FN = ((vote_pred == 0) & (targetlist == 1)).sum()
            FP = ((vote_pred == 1) & (targetlist == 0)).sum()

            print('TP=', TP, 'TN=', TN, 'FN=', FN, 'FP=', FP)
            print('TP+FP', TP + FP)
            p = TP / (TP + FP)
            print('precision', p)
            p = TP / (TP + FP)
            r = TP / (TP + FN)
            print('recall', r)
            F1 = 2 * r * p / (r + p)
            acc = (TP + TN) / (TP + TN + FP + FN)
            print('F1', F1)
            print('acc', acc)
            AUC = roc_auc_score(targetlist, vote_score)
            print('AUC', AUC)

            # 训练过程可视化
            train_loss = train_loss.cpu().detach().numpy()
            val_loss = val_loss.cpu().detach().numpy()
            viz.line([[p, r, AUC, F1, acc]], [epoch], win='train_performance', update='append',
                     opts=dict(title='train_performance', legend=['precision', 'recall', 'AUC', 'F1', 'acc']))
            viz.line([[train_loss], [val_loss]], [epoch], win='train_Loss', update='append',
                     opts=dict(title='train_Loss', legend=['train_loss', 'val_loss']))

            print(
                '\n The epoch is {}, average recall: {:.4f}, average precision: {:.4f},average F1: {:.4f}, '
                'average accuracy: {:.4f}, average AUC: {:.4f}'.format(
                    epoch, r, p, F1, acc, AUC))

            # 更新模型

            if os.path.exists('backup') == 0:
                os.makedirs('backup')
            torch.save(model.state_dict(), "backup/{}.pt".format(modelname))

            vote_pred = np.zeros(valset.__len__())
            vote_score = np.zeros(valset.__len__())
            f = open('performance/{}.txt'.format(modelname), 'a+')
            f.write(
                '\n The epoch is {}, average recall: {:.4f}, average precision: {:.4f},average F1: {:.4f}, '
                'average accuracy: {:.4f}, average AUC: {:.4f}'.format(
                    epoch, r, p, F1, acc, AUC))
            f.close()
        if epoch % (votenum*10) == 0:  # 每100个epoch，保存一次模型
            torch.save(model.state_dict(), "backup/{}_epoch{}.pt".format(modelname, epoch))
