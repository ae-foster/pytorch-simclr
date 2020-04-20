'''This script trains the downstream classifier using gradients (for large datasets).'''
import argparse
import os

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from configs import get_datasets
from evaluate.lbfgs import test
from models import *

parser = argparse.ArgumentParser(description='Train downstream classifier with gradients.')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument("--momentum", default=0.9, type=float, help='SGD momentum')
parser.add_argument("--batch-size", type=int, default=512, help='Training batch size')
parser.add_argument("--num-epochs", type=int, default=90, help='Number of training epochs')
parser.add_argument("--num-workers", type=int, default=2, help='Number of threads for data loaders')
parser.add_argument("--weight-decay", type=float, default=1e-6, help='Weight decay on the linear classifier')
parser.add_argument("--nesterov", action="store_true", help="Turn on Nesterov style momentum")
parser.add_argument("--load-from", type=str, default='ckpt.pth', help='File to load from')
args = parser.parse_args()

# Load checkpoint.
print('==> Loading settings from checkpoint..')
assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
resume_from = os.path.join('./checkpoint', args.load_from)
checkpoint = torch.load(resume_from)
args.dataset = checkpoint['args']['dataset']
args.arch = checkpoint['args']['arch']

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0

# Data
print('==> Preparing data..')
_, testset, clftrainset, num_classes, stem = get_datasets(args.dataset, augment_clf_train=True)

testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False,
                                         num_workers=args.num_workers, pin_memory=True)
clftrainloader = torch.utils.data.DataLoader(clftrainset, batch_size=args.batch_size, shuffle=True,
                                             num_workers=args.num_workers, pin_memory=True)

# Model
print('==> Building model..')
##############################################################
# Encoder
##############################################################
if args.arch == 'resnet18':
    net = ResNet18(stem=stem)
elif args.arch == 'resnet34':
    net = ResNet34(stem=stem)
elif args.arch == 'resnet50':
    net = ResNet50(stem=stem)
else:
    raise ValueError("Bad architecture specification")
net = net.to(device)

##############################################################
# Classifier
##############################################################
clf = nn.Linear(net.representation_dim, num_classes).to(device)

if device == 'cuda':
    repr_dim = net.representation_dim
    net = torch.nn.DataParallel(net)
    net.representation_dim = repr_dim
    cudnn.benchmark = True

print('==> Loading encoder from checkpoint..')
net.load_state_dict(checkpoint['net'])

criterion = nn.CrossEntropyLoss()
clf_optimizer = optim.SGD(clf.parameters(), lr=args.lr, momentum=args.momentum, nesterov=args.nesterov,
                          weight_decay=args.weight_decay)


def train_clf(epoch):
    print('\nEpoch %d' % epoch)
    net.eval()
    clf.train()
    train_loss = 0
    t = tqdm(enumerate(clftrainloader), desc='Loss: **** ', total=len(clftrainloader), bar_format='{desc}{bar}{r_bar}')
    for batch_idx, (inputs, targets) in t:
        clf_optimizer.zero_grad()
        inputs, targets = inputs.to(device), targets.to(device)
        representation = net(inputs).detach()
        predictions = clf(representation)
        loss = criterion(predictions, targets)
        loss.backward()
        clf_optimizer.step()

        train_loss += loss.item()

        t.set_description('Loss: %.3f ' % (train_loss / (batch_idx + 1)))


for epoch in range(args.num_epochs):
    train_clf(epoch)
    acc = test(testloader, device, net, clf)
    if acc > best_acc:
        best_acc = acc
print("Best test accuracy", best_acc, "%")
