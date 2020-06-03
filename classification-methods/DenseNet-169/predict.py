import os
import csv
import torch
import warnings
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torchxrayvision as xrv
from torchvision import transforms
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

from tools.conduct import test
from tools.dataload import CovidCTDataset

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 依通道标准化

test_transformer = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize
])

if __name__ == '__main__':

    batchsize = 24  # 原来用的10，这里改成24，根据GPU容量确度。

    # ------------------------------- step 1/3 数据 ----------------------------

    # 实例化CovidCTDataset
    testset = CovidCTDataset(root_dir='C:/Users/LZD/Desktop/work/COVID-CT-master/Data',
                             txt_COVID='C:/Users/LZD/Desktop/work/COVID-CT-master/Data/testCT_COVID.txt',
                             txt_NonCOVID='C:/Users/LZD/Desktop/work/COVID-CT-master/Data/testCT_NonCOVID.txt',
                             transform=test_transformer)
    print(testset.__len__())

    # 构建DataLoader

    test_loader = DataLoader(testset, batch_size=batchsize, drop_last=False, shuffle=False)

    # ------------------------------ step 2/3 模型 --------------------------------

    model = xrv.models.DenseNet(num_classes=2, in_channels=3).cuda()  # DenseNet 模型，二分类
    modelname = 'DenseNet_medical'
    torch.cuda.empty_cache()

    # ----------------------------  step 3/3 测试 ------------------------------

    f = open(f'performance/test_model.csv', mode='w')
    csv_writer = csv.writer(f)
    flag = 1

    for modelname in os.listdir('backup'):
        model.load_state_dict(torch.load('backup/{}'.format(modelname)))
        torch.cuda.empty_cache()

        bs = 10

        warnings.filterwarnings('ignore')

        r_list = []
        p_list = []
        acc_list = []
        AUC_list = []
        TP = 0
        TN = 0
        FN = 0
        FP = 0

        vote_score = np.zeros(testset.__len__())

        targetlist, scorelist, predlist = test(model, test_loader)
        vote_score = vote_score + scorelist

        TP = ((predlist == 1) & (targetlist == 1)).sum()
        TN = ((predlist == 0) & (targetlist == 0)).sum()
        FN = ((predlist == 0) & (targetlist == 1)).sum()
        FP = ((predlist == 1) & (targetlist == 0)).sum()

        p = TP / (TP + FP)
        p = TP / (TP + FP)
        r = TP / (TP + FN)
        F1 = 2 * r * p / (r + p)
        acc = (TP + TN) / (TP + TN + FP + FN)

        AUC = roc_auc_score(targetlist, vote_score)

        print(
            '\n{}, recall: {:.4f}, precision: {:.4f},F1: {:.4f}, accuracy: {:.4f}, AUC: {:.4f}'.format(
                modelname, r, p, F1, acc, AUC))
        if flag:
            header = ['modelname', 'recall', 'precision', 'F1', 'accuracy', 'AUC']
            csv_writer.writerow(header)
            flag = 0
        row = [modelname, str(r), str(p), str(F1), str(acc), str(AUC)]
        csv_writer.writerow(row)

    f.close()
