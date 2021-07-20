from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

import logging
import time
import models

from vgg import *
#import models
from lowrank_vgg import LowRankVGG, FullRankVGG
from resnet_cifar10 import *

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))


def add_weight_decay(model, weight_decay=1e-5, skip_list=()):
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if len(param.shape) == 1 or name in skip_list:
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {'params': no_decay, 'weight_decay': 0.},
        {'params': decay, 'weight_decay': weight_decay}]


def load_checkpoint(model, resume_epoch, device):
    with open("ckpt_model_resnet18_epoch{}".format(resume_epoch), "rb") as model_checkpoint:
        resnet18_epoch10 = torch.load(model_checkpoint)
    # load desired weights to the model
    new_state_dict = {}
    model_counter_ = 0
    for param_idx,(key_name, param) in enumerate(model.state_dict().items()):
        if key_name in resnet18_epoch10.keys():
            new_state_dict[key_name] = resnet18_epoch10[key_name]
        else:
            new_state_dict[key_name] = param
    model.load_state_dict(new_state_dict)
    model.to(device)
    logger.info("Loading successfully ...")


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=300, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--resume-epoch', type=int, default=10, metavar='RE',
                        help='starting from which epoch')
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Benchmarking over device: {}".format(device))

    normalize = transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
                                std=[x/255.0 for x in [63.0, 62.1, 66.7]])
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: F.pad(
                            Variable(x.unsqueeze(0), requires_grad=False),
                            (4,4,4,4),mode='reflect').data.squeeze()),
        transforms.ToPILImage(),
        transforms.RandomCrop(32),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
        ])
    # data prep for test set
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        normalize])
    # load training and test set here:
    training_set = datasets.CIFAR10(root='./cifar10_data', train=True,
                                            download=True, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(training_set, batch_size=args.batch_size,
                                              shuffle=True)
    testset = datasets.CIFAR10(root='./cifar10_data', train=False,
                                           download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size,
                                             shuffle=False)


    model = LowRankResNet18().to(device)
    #model = BaselineResNet18().to(device)

    #model = LowRankResNet18LR().to(device)
    logger.info("============> Model info: {}".format(model))

    load_checkpoint(model, args.resume_epoch, device)

    # eval
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(F.log_softmax(output), target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    logger.info('\nEpoch: {}, Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(args.resume_epoch, 
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    #model = FullRankVGG().to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=1e-4)
    
    # switching off the weight decay for batch norm layers
    #parameters = add_weight_decay(model, 0.0001)
    #weight_decay = 0.
    # optimizer = torch.optim.SGD(parameters, args.lr,
    #                             momentum=args.momentum,
    #                             #weight_decay=args.weight_decay)
    #                             weight_decay=weight_decay)

    #optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
    criterion = nn.CrossEntropyLoss()
    init_lr = args.lr

    for epoch in range(args.resume_epoch, args.epochs + 1):
        # adjusting lr schedule
        if epoch < 150:
            for group in optimizer.param_groups:
                group['lr'] = init_lr
        elif (epoch >= 150 and epoch < 250):
            for group in optimizer.param_groups:
                group['lr'] = init_lr/10.0
        elif epoch >= 250:
            for group in optimizer.param_groups:
                group['lr'] = init_lr/100.0

        for group in optimizer.param_groups:
            logger.info("Epoch: {}, Current effective lr: {}".format(epoch, group['lr']))

        # train
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()

            #torch.cuda.synchronize()
            #iter_comp_start = time.time()
            output = model(data)

            loss = criterion(output, target)
            #torch.cuda.synchronize()
            #forward_dur = time.time() - iter_comp_start

            #torch.cuda.synchronize()
            #backward_start = time.time()
            loss.backward()
            #torch.cuda.synchronize()
            #backward_dur = time.time() - backward_start

            optimizer.step()

            #torch.cuda.synchronize()
            #iter_comp_dur = time.time() - iter_comp_start

            logger.info('Train Epoch: {} [{}/{} ({:.0f}%)]  Loss: {:.6f}'.format(
               epoch, batch_idx * len(data), len(train_loader.dataset),
               100. * batch_idx / len(train_loader), loss.item()))

        # eval
        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += F.nll_loss(F.log_softmax(output), target, reduction='sum').item()  # sum up batch loss
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)
        logger.info('\nEpoch: {}, Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(epoch, 
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))

    # we save the final model for future use
    with open("trained_model_resnet18", "wb") as f_:
        torch.save(model.state_dict(), f_)

if __name__ == '__main__':
    main()