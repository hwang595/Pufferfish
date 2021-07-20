import os
import random
import shutil
import time
import warnings

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
# import torchvision.models as models
from . import models

class imagenetTrain(object):
    def __init__(self, model_config):
        self.device = model_config['device']
        if model_config['early_bird']:
            self.model = self._create_model(model_config['arch'],
                                        False,
                                        model_config['rank_factor'], 
                                        scratch_dir=model_config['scratch'])            
        else:
            # vanilla model
            self.model = self._create_model(model_config['arch'],
                                        is_lowrank=False,
                                        rank_factor=model_config['rank_factor'])

            if model_config['training_mode'] in ("pufferfish", "powerfish"):
                self.rank_factor = model_config['rank_factor']
                self.lowrank_model = self._create_model(model_config['lowrank_arch'],
                                                    is_lowrank=True,
                                                    rank_factor=model_config['rank_factor'])

        # full train loader doesn't do sampling
        self.train_loader, self.test_loader, self.sampler = self._create_data_loader(
            model_config['data_path'], model_config['num_dataloader_threads'],
        model_config['train_batch_size'], model_config['test_batch_size'],
            model_config['is_distributed'])

        self.training_mode = model_config["training_mode"]
        self.parameter_names = [name for (name, _) in self.model.named_parameters()]
        self.criterion = LabelSmoothingLoss(classes=1000,
                                            smoothing=0.1).to(self.device)
        self.lr = model_config['init_lr']

        if model_config['training_mode'] in ("pufferfish", "powerfish"):
            self.full_rank_warmup_epoch = model_config["full_rank_warmup_epoch"]
            self.lowrank_parameter_names = [name for (name, _) in self.lowrank_model.named_parameters()]

    def _create_model(self, arch, is_lowrank, rank_factor, scratch_dir=None):
        if is_lowrank:
            model = models.__dict__[arch](rank_factor=rank_factor)
            model.to(self.device)
        elif arch == "resnet50_prune":
            checkpoint = torch.load(scratch_dir, map_location=torch.device("cpu"))
            cfg_input = checkpoint['cfg']
            model = models.__dict__[arch](pretrained=False, cfg=cfg_input)
            model.to(self.device)
        else:
            model = models.__dict__[arch]()
            model.to(self.device)
        return(model)

    def _create_data_loader(self, data_path, num_workers, train_batch_size,
                            test_batch_size, is_distributed):
        """
        Returns test and train loaders for a given dataset
        
        """
        train_dir = os.path.join(data_path, 'train')
        val_dir = os.path.join(data_path, "val")

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        train_dataset = datasets.ImageFolder(
            train_dir,
            transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]))

        sampler = None
        is_shuffle = True
        if is_distributed:
            sampler = torch.utils.data.DistributedSampler(train_dataset)
            is_shuffle = False
        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=train_batch_size,
                                                   shuffle=is_shuffle,
                                                   num_workers=num_workers,
                                                   pin_memory=True,
                                                   sampler=sampler)

        test_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(val_dir, transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
        ])),
            batch_size=test_batch_size, 
            shuffle=False,
            num_workers=num_workers, 
            pin_memory=True)

        return (train_loader, test_loader, sampler)

    def train_single_iter(self, epoch=None, logger=None, for_autoscale=False):
        """
        Train single iter
        """ 
        for batch_idx, (data, target) in enumerate(train_data_loader):
            data, target = data.to(self.device), target.to(self.device)
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            grad_array = [param.grad.data for param in self.model.parameters()]
            if batch_idx%100 == 0:
                if logger is not None:

                    logger.info('Train Epoch(imagenet): {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                                    epoch, batch_idx, len(train_data_loader),
                                    100. * batch_idx / len(self.train_loader), loss.item()))
            yield grad_array

    def batch_loss_and_gradient(self, batch_idx, data, target, logger=None, epoch=None):
        data, target = data.to(self.device), target.to(self.device)
        #output = self.model(data)
        if self.training_mode == "vanilla":
            output = self.model(data)
        elif self.training_mode in ("pufferfish", "powerfish"):
            if epoch in range(self.full_rank_warmup_epoch):
                output = self.model(data)
            else:
                output = self.lowrank_model(data)
        else:
            raise NotImplementedError("Unsupported training mode !!!!")

        loss = self.criterion(output, target)
        loss.backward()
        #grad_array = [param.grad.data for param in self.model.parameters()]
        
        if self.training_mode == "vanilla":
            grad_array = [param.grad.data for param in self.model.parameters()]
        elif self.training_mode  in ("pufferfish", "powerfish"):
            if epoch in range(self.full_rank_warmup_epoch):
                grad_array = [param.grad.data for param in self.model.parameters()]
            else:
                grad_array = [param.grad.data for param in self.lowrank_model.parameters()]
        else:
            raise NotImplementedError("Unsupported training mode !!!!")

        if batch_idx%100 == 0:
            if logger is not None:
                logger.info('Train Epoch(imagenet): {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                                epoch, batch_idx, len(self.train_loader),
                                100. * batch_idx / len(self.train_loader), loss.item()))
        return grad_array

    
    def validate_model(self, logger):
        self.model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.test_loader):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                test_loss += self.criterion(output, target).item()
                pred = output.max(1, keepdim=True)[1]
                correct += pred.eq(target.view_as(pred)).sum().item()
        test_loss /= len(self.test_loader)
        logger.info('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
                test_loss, correct, len(self.test_loader.dataset),
                100. * correct / len(self.test_loader.dataset)))
        self.model.train()
        return test_loss


    def validate_lowrank_model(self, logger):
        self.lowrank_model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.test_loader):
                data, target = data.to(self.device), target.to(self.device)
                output = self.lowrank_model(data)
                test_loss += self.criterion(output, target).item()
                pred = output.max(1, keepdim=True)[1]
                correct += pred.eq(target.view_as(pred)).sum().item()
        test_loss /= len(self.test_loader)
        logger.info('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
                test_loss, correct, len(self.test_loader.dataset),
                100. * correct / len(self.test_loader.dataset)))
        self.lowrank_model.train()
        return test_loss

    
class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes, smoothing=0.0, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            # true_dist = pred.data.clone()
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))
