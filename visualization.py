# -*- coding: utf-8 -*-
# @Time : 2020/4/12 10:32
# @Author : Ruiqi Wang

from __future__ import print_function

import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from matplotlib import pyplot as plt

import random
from copy import deepcopy
import itertools

DATA_DIR = './DATA'

l2c = [(i, i) for i in range(10)] + list(itertools.permutations(range(10), 2))
VRR2label = dict(zip(l2c, list(range(len(l2c)))))


def VRR(data, label, train=True):
    '''

    Args:
        data: N,1,28,28
        label: N

    Returns:

    '''
    data_up, data_bottom = data.split(14, dim=2)

    indicate = list(range(label.size(0)))
    random.shuffle(indicate)
    indicate_up = deepcopy(indicate)
    random.shuffle(indicate)
    indicate_bottom = deepcopy(indicate)

    data_VRR = torch.cat([data_up[indicate_up], data_bottom[indicate_bottom]], dim=2)
    label_VRR = list(zip(label[indicate_up].tolist(), label[indicate_bottom].tolist()))

    label = list(zip(label.tolist(), label.tolist()))
    # print(label_VRR)
    # exit()

    if train:
        data = torch.cat([data, data_VRR], dim=0)
        label = label + label_VRR

    label = [VRR2label[i] for i in label]
    label = torch.Tensor(label).to(data.device).long()

    return data, label


class Net(nn.Module):
    def __init__(self, num_classes=10, feature_length=2):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc1 = nn.Sequential(
            nn.Linear(3 * 3 * 128, feature_length),
        )
        # self.fc2 = nn.Linear(feature_length, num_classes)
        self.centers = nn.Parameter(torch.rand(num_classes, feature_length))

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)
        features = self.fc1(x)
        out = self.distance(features, self.centers)
        return out, features, self.centers

    def distance(self, features, centers):
        f_2 = features.pow(2).sum(dim=1, keepdim=True)
        c_2 = centers.pow(2).sum(dim=1, keepdim=True)
        dist = f_2 - 2 * torch.matmul(features, centers.transpose(0, 1)) + c_2.transpose(0, 1)
        return dist


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        idx = torch.where(target < 6)
        data = data[idx]
        target = target[idx]

        # data, target = VRR(data, target)

        optimizer.zero_grad()
        output, _, _ = model(data)
        # print(output)
        loss = F.cross_entropy(-output, target)
        # print(loss)
        loss.backward()
        optimizer.step()
        # if batch_idx == 2:
        #     exit()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    all = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)

            idx = torch.where(target < 6)
            data = data[idx]
            target = target[idx]

            # data, target = VRR(data, target, False)

            output, _, _ = model(data)
            test_loss += F.nll_loss(-output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmin(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            all += len(idx[0])

    # test_loss /= len(test_loader.dataset)
    test_loss /= all

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, all, 100. * correct / all))


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current Model')
    parser.add_argument('--split', type=int, default=6)

    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    global VRR2label
    l2c = [(i, i) for i in range(args.split)] + list(itertools.permutations(range(args.split), 2))
    VRR2label = dict(zip(l2c, list(range(len(l2c)))))

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(root=DATA_DIR, train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(root=DATA_DIR, train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)

    model = Net(num_classes=args.split).to(device)
    # model = Net(num_classes=args.split ** 2).to(device)
    # optimizer = optim.Adadelta(model.parameters(), lr=0.01)
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)
        scheduler.step()

    # if args.save_model:
    torch.save(model.state_dict(), "mnist_cnn_{}-noVRR.pt".format(args.split))


def visulize(data_loader, use_cuda=True):
    device = torch.device("cuda" if use_cuda else "cpu")
    model = Net(num_classes=36).to(device)
    model.load_state_dict(torch.load('mnist_cnn_6.pt'))
    # model = Net(num_classes=6).to(device)
    # model.load_state_dict(torch.load('mnist_cnn_6-noVRR.pt'))

    features = []
    labels = []

    with torch.no_grad():
        for data, target in data_loader:
            # print(data.size(), target.size())
            data, target = data.to(device), target.to(device)

            # idx = torch.where(target < 6)
            # data = data[idx]
            # target = target[idx]

            output, feature, center = model(data)
            # test_loss += F.nll_loss(-output, target, reduction='sum').item()  # sum up batch loss
            # pred = output.argmin(dim=1, keepdim=True)  # get the index of the max log-probability
            # correct += pred.eq(target.view_as(pred)).sum().item()
            features.append(feature.cpu())
            labels.append(target.cpu())

    features = torch.cat(features).numpy()
    # print(labels[0])
    labels = torch.cat(labels).numpy()

    fig = plt.figure(figsize=(20, 20))
    # plt.axis('off')
    plt.xticks([])
    plt.yticks([])
    ax1 = fig.add_subplot(111)
    # ax1.set_title('Scatter Plot')
    # plt.xlabel('X')
    # plt.ylabel('Y')
    # cValue = ['c', 'orange', 'g', 'cyan', 'r', 'y', 'gray', 'purple', 'black', 'b']
    cValue = ['c', 'orange', 'g', 'gray', 'r', 'y', 'cyan', 'purple', 'black', 'b']

    cs = [cValue[i] for i in labels]
    ax1.scatter(features[:, 0], features[:, 1], s=100, c=cs, marker='.')

    plt.show()
    plt.savefig("100VAC10.pdf")


if __name__ == '__main__':
    # main()

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(DATA_DIR, train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=1000, shuffle=True)

    visulize(test_loader)


