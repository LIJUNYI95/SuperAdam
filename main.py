'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
# from SuperAdam_new import SuperAdam
from SuperAdam import SuperAdam
# from SAdam import SuperAdam
from storm import Storm
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from models import *
from utils import progress_bar

import pdb
import numpy as np

# os.environ["CUDA_VISIBLE_DEVICES"] = '2'

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=3, type=float, help='learning rate')
parser.add_argument('--m', default=2400, type=float, help='w')
parser.add_argument('--c', default=2, type=float, help='c')
parser.add_argument('--gamma', default=0.03, type=float, help='gamma')
parser.add_argument('--beta', default=0.999, type=float, help='beta')
parser.add_argument('--beta1', default=0.9, type=float, help='beta1')
# parser.add_argument('--warm_iters', default=75, type=int, help='number of iters to use adam')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
args = parser.parse_args()
print(args.lr, args.m, args.c, args.gamma)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

# Model
print('==> Building model..')
# net = VGG('VGG19')
net = ResNet18()
# net = PreActResNet18()
# net = GoogLeNet()
# net = DenseNet121()
# net = ResNeXt29_2x64d()
# net = MobileNet()
# net = MobileNetV2()
# net = DPN92()
# net = ShuffleNetG2()
# net = SENet18()
# net = ShuffleNetV2(1)
# net = EfficientNetB0()
# net = RegNetX_200MF()
# net = SimpleDLA()
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

global_size = True
coordinate_global_size = False
b_b = False
use_adam = True
# tau = True

criterion = nn.CrossEntropyLoss()
optimizer = SuperAdam(net.parameters(), lr=args.lr, c=args.c, m=args.m, gamma=args.gamma, beta=args.beta, beta1=args.beta1,\
    amsgrad=True, global_size=global_size, coordinate_global_size=coordinate_global_size,\
        b_b = b_b, use_adam=use_adam)
# optimizer = SuperAdam(net.parameters(), tau=tau)
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=300)

old_state_dict = {}
acc_save = []; train_acc_save = []
loss_save = []; train_loss_save = []
g_norm_save = []; c_num_save = []
# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    global old_state_dict
    g_norm = []; c_num = []
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)

        if not use_adam:
            cur_state_dict = {}
            for k, v in net.state_dict().items():
                cur_state_dict[k] = v.data.clone()

            if old_state_dict:
                net.load_state_dict(old_state_dict)
                optimizer.zero_grad()
                outputs = net(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.update_momentum()

            #only for superadam
            net.load_state_dict(cur_state_dict)
            old_state_dict = {}
            for k, v in cur_state_dict.items():
                old_state_dict[k] = v.data.clone()

            
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        h_max, h_min, gg = optimizer.step()
        g_norm.append(gg); c_num.append(h_max/h_min) 

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()      

        progress_bar(batch_idx, len(trainloader), 'Loss: %.5f | Acc: %.5f%% (%d/%d) | G_Norm: %.5f | C_num: %.5f'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total, gg, h_max/h_min))
    
    acc = 100. * correct/total
    train_acc_save.append(acc)
    train_loss_save.append(train_loss/(batch_idx + 1))
    g_norm_save.append(np.mean(g_norm))
    c_num_save.append(np.mean(c_num))


def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            # print(loss.item())
            # pdb.set_trace()
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.5f | Acc: %.5f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    acc_save.append(acc)
    loss_save.append(test_loss/(batch_idx + 1))
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.pth')
        best_acc = acc


dire = 'SuperAdam_results/'
postfix = '-' + str(args.lr) + '-' + str(args.m) + '-' + str(args.c) + '-' + str(args.gamma) + '-' + str(args.beta)

prefix = ''
if global_size:
    prefix = '_global_coordinate'
elif coordinate_global_size:
    prefix = '_coordinate_global'
elif b_b:
    prefix = '_b_b'
else:
    prefix = '_super_adam'
    
if use_adam:
    prefix += '_adam_like'
postfix = prefix + postfix

# postfix = ''

for epoch in range(start_epoch, start_epoch+201):
    train(epoch)
    test(epoch)
    # pdb.set_trace()
    # scheduler.step()

    # if epoch == 75: 
    #     for group in optimizer.param_groups:
    #         group['beta'] = 0.5
    #         # group['gamma'] /= 2
    # if epoch == 125: 
    #     for group in optimizer.param_groups:
    #         group['beta'] = 0.3
    # if epoch == 175: 
    #     for group in optimizer.param_groups:
    #         group['beta'] *= 0.5


    if epoch % 10 == 0:        
        np.save(dire + 'acc' + postfix, acc_save)
        np.save(dire + 'train_acc' + postfix, train_acc_save)
        np.save(dire + 'loss' + postfix, loss_save)
        np.save(dire + 'train_loss' + postfix, train_loss_save)
        np.save(dire + 'g_norm' + postfix, g_norm_save)
        np.save(dire + 'c_num' + postfix, c_num_save)

    if epoch == 100: 
        for group in optimizer.param_groups:
            # group['betas'] = (0.99, 0.999)
            group['lr'] /= 2
    if epoch == 150: 
        for group in optimizer.param_groups:
            group['lr'] /= 2
