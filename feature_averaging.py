'''Train CIFAR10/100 with PyTorch using standard Contrastive Learning. This script tunes the L2 reg weight of the
final classifier.'''
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn

import math
import os
import argparse

from models import *
from configs import get_datasets
from evaluate import train_clf, encode_feature_averaging

parser = argparse.ArgumentParser(description='Final evaluation with feature averaging.')
parser.add_argument("--num-workers", type=int, default=2, help='Number of threads for data loaders')
parser.add_argument("--load-from", type=str, default='ckpt.pth', help='File to load from')
parser.add_argument("--num-passes", type=int, default=10, help='Number of passes to average')
parser.add_argument("--reg-lower", type=float, default=-7, help='Minimum log regularization parameter (base 10)')
parser.add_argument("--reg-upper", type=float, default=-3, help='Maximum log regularization parameter (base 10)')
parser.add_argument("--num-steps", type=int, default=10, help='Number of log-linearly spaced reg parameters to try')
args = parser.parse_args()

# Load checkpoint.
print('==> Loading settings from checkpoint..')
assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
resume_from = os.path.join('./checkpoint', args.load_from)
checkpoint = torch.load(resume_from)
args.dataset = checkpoint['args']['dataset']
args.arch = checkpoint['args']['arch']

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Data
print('==> Preparing data..')
_, testset, clftrainset, num_classes, stem = get_datasets(args.dataset, test_as_train=True)

testloader = torch.utils.data.DataLoader(testset, batch_size=1000, shuffle=False, num_workers=args.num_workers,
                                         pin_memory=True)
clftrainloader = torch.utils.data.DataLoader(clftrainset, batch_size=1000, shuffle=False, num_workers=args.num_workers,
                                             pin_memory=True)

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

if device == 'cuda':
    repr_dim = net.representation_dim
    net = torch.nn.DataParallel(net)
    net.representation_dim = repr_dim
    cudnn.benchmark = True

print('==> Loading encoder from checkpoint..')
net.load_state_dict(checkpoint['net'])


best_acc = 0
X, y = encode_feature_averaging(clftrainloader, device, net, num_passes=args.num_passes)
X_test, y_test = encode_feature_averaging(testloader, device, net, num_passes=args.num_passes)
for reg_weight in torch.exp(math.log(10) * torch.linspace(args.reg_lower, args.reg_upper, args.num_steps,
                                                          dtype=torch.float, device=device)):
    clf = train_clf(X, y, net.representation_dim, num_classes, device, reg_weight=reg_weight)
    raw_scores = clf(X_test)
    _, predicted = raw_scores.max(1)
    correct = predicted.eq(y_test).sum().item()
    acc = 100 * correct / predicted.shape[0]
    print('Test accuracy', acc, '%')
    if acc > best_acc:
        best_acc = acc
print("Best test accuracy", best_acc, "%")
