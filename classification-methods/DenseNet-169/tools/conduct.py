import os
import torch
import numpy as np
import torch.nn.functional as F

device = 'cuda'


def train(optimizer, epoch, model, train_loader, modelname, criteria):
    model.train()  # 训练模式
    bs = 24
    train_loss = 0
    train_correct = 0

    for batch_index, batch_samples in enumerate(train_loader):

        # move data to device
        data, target = batch_samples['img'].to(device), batch_samples['label'].to(device)
        # data形状，torch.Size([32, 3, 224, 224])
        # data = data[:, 0, :, :]  # 原作者只取了第一个通道的数据来训练，笔者改成了3个通道

        # data = data[:, None, :, :]
        # data形状，torch.Size([32, 1, 224, 224])

        optimizer.zero_grad()

        output = model(data)
        loss = criteria(output, target.long())
        train_loss += criteria(output, target.long())  # 后面求平均误差用的

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pred = output.argmax(dim=1, keepdim=True)
        train_correct += pred.eq(target.long().view_as(pred)).sum().item()  # 累加预测与标签吻合的次数，用于后面算准确率

        # 显示一个epoch的进度，425张图片，批大小是24，一个epoch需要18次迭代
        if batch_index % 4 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tTrain Loss: {:.6f}'.format(
                epoch, batch_index, len(train_loader),
                100.0 * batch_index / len(train_loader), loss.item() / bs))
    # print(len(train_loader.dataset))   # 425
    print('\nTrain set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        train_loss / len(train_loader.dataset), train_correct, len(train_loader.dataset),
        100.0 * train_correct / len(train_loader.dataset)))

    if os.path.exists('performance') == 0:
        os.makedirs('performance')
    f = open('performance/{}.txt'.format(modelname), 'a+')
    f.write('\nTrain set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        train_loss / len(train_loader.dataset), train_correct, len(train_loader.dataset),
        100.0 * train_correct / len(train_loader.dataset)))
    f.write('\n')
    f.close()

    return train_loss / len(train_loader.dataset)  # 返回一个epoch的平均误差，用于可视化损失
def val(model, val_loader, criteria):

    model.eval()
    val_loss = 0
    correct = 0

    # Don't update model
    with torch.no_grad():

        predlist = []
        scorelist = []
        targetlist = []
        # Predict
        for batch_index, batch_samples in enumerate(val_loader):
            data, target = batch_samples['img'].to(device), batch_samples['label'].to(device)
            # data = data[:, 0, :, :]  # 原作者只取了第一个通道的数据，笔者改成了3个通道

            # data = data[:, None, :, :]
            # data形状，torch.Size([32, 1, 224, 224])
            output = model(data)

            val_loss += criteria(output, target.long())
            score = F.softmax(output, dim=1)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.long().view_as(pred)).sum().item()

            targetcpu = target.long().cpu().numpy()    # 由GPU->CPU
            predlist = np.append(predlist, pred.cpu().numpy())
            scorelist = np.append(scorelist, score.cpu().numpy()[:, 1])
            targetlist = np.append(targetlist, targetcpu)

    return targetlist, scorelist, predlist, val_loss / len(val_loader.dataset)
def test(model, test_loader):
    model.eval()

    # Don't update model
    with torch.no_grad():

        predlist = []
        scorelist = []
        targetlist = []
        # Predict
        for batch_index, batch_samples in enumerate(test_loader):
            data, target = batch_samples['img'].to(device), batch_samples['label'].to(device)
            # data = data[:, 0, :, :]            #  只取了第一个通道的数据来训练，笔者改成了灰度图像

            # data = 0.299 * data[:, 0, :, :] + 0.587 * data[:, 1, :, :] + 0.114 * data[:, 2, :, :]
            # data形状，torch.Size([32, 224, 224])

            # data = data[:, None, :, :]
            # data形状，torch.Size([32, 1, 224, 224])
            #             print(target)
            output = model(data)
            score = F.softmax(output, dim=1)
            pred = output.argmax(dim=1, keepdim=True)

            targetcpu = target.long().cpu().numpy()
            predlist = np.append(predlist, pred.cpu().numpy())
            scorelist = np.append(scorelist, score.cpu().numpy()[:, 1])
            targetlist = np.append(targetlist, targetcpu)

    return targetlist, scorelist, predlist
