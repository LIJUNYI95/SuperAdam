'''Train CIFAR10 with SuperAdam.'''
import os
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms

from models import *
from utils import progress_bar
from SuperAdam import SuperAdam

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--k', default=3, type=float, help='scaling coefficient of mu')
parser.add_argument('--m', default=2400, type=float, help='offset coefficient of mu')
parser.add_argument('--c', default=40, type=float, help='coefficient of the momentum')
parser.add_argument('--gamma', default=0.003, type=float, help='scaling coefficient of the quadratic optimization problem')
parser.add_argument('--beta', default=0.999, type=float, help='exponential average coefficient')
parser.add_argument('--tau', dest='tau', action='store_false', help='if set True: use storm-like variance reduction update, \
    otherwise: use adam-like momentum update')
parser.add_argument('--coord_glob_H', dest='coord_glob_H', action='store_true', help='alternative choice of H')
parser.add_argument('--glob_H', dest='glob_H', action='store_true', help='alternative choice of H')


args = parser.parse_args()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0
start_epoch = 0


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

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Model
print('==> Building model..')
net = ResNet18()
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

criterion = nn.CrossEntropyLoss()
optimizer = SuperAdam(net.parameters(), k=args.k, c=args.c, m=args.m, gamma=args.gamma, beta=args.beta, \
    glob_H=args.glob_H, coord_glob_H=args.coord_glob_H, tau=args.tau)

old_state_dict = {}
acc_save = []; train_acc_save = []
loss_save = []; train_loss_save = []
# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    global old_state_dict
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)

        if args.tau:
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
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()      

        progress_bar(batch_idx, len(trainloader), 'Loss: %.5f | Acc: %.5f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    
    acc = 100. * correct/total
    train_acc_save.append(acc)
    train_loss_save.append(train_loss/(batch_idx + 1))

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
if not os.path.isdir('SuperAdam_results'):
    os.mkdir('SuperAdam_results')
postfix = '-' + str(args.k) + '-' + str(args.m) + '-' + str(args.c) + '-' + str(args.gamma) + '-' + str(args.beta)

prefix = ''
if args.glob_H:
    prefix = '_glob_H'
elif args.coord_glob_H:
    prefix = '_coord_glob_H'
else:
    prefix = '_super_adam'
    
if args.tau:
    prefix += '_adam_like'
postfix = prefix + postfix

for epoch in range(start_epoch, start_epoch+201):
    train(epoch)
    test(epoch)

    if epoch % 10 == 0:        
        np.save(dire + 'acc' + postfix, acc_save)
        np.save(dire + 'train_acc' + postfix, train_acc_save)
        np.save(dire + 'loss' + postfix, loss_save)
        np.save(dire + 'train_loss' + postfix, train_loss_save)

    if epoch == 100 or epoch == 150: 
        for group in optimizer.param_groups:
            group['k'] /= 2
