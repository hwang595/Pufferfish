import os
import time
import json
import math
import random
import datetime
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.optim as optim
from torchvision import datasets, transforms
from collections import defaultdict
import argparse
import logging

from torch.autograd import Variable
from gradient_reducers import StochasticUniformQuantization, SignSGDwithMajorityVoteReducer, RankKReducer

from torch.optim import lr_scheduler
from warmup_scheduler import GradualWarmupScheduler


from utils import *

# added files
# import grad_utils
import train_network
import sparsify_gradient
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors


imagenet_vanilla_config = {
    "name" : "imagenet",
    "arch" : "resnet50",
    "is_lowrank": False,
    "rank_factor": 4,
    "dataset" : "imagenet",
    "weight_decay": 0.0001,
    "optimizer_weight_decay_conv":0.0001,
    "optimizer_weight_decay_other":0.0001,
    "optimizer_weight_decay_bn":0.0,
    "device" : "cuda:0",
    "data_path": "/home/ubuntu/data",
    "num_dataloader_threads": 8,
    "train_batch_size": 16,
    "test_batch_size": 128,
    "init_lr": 0.0001,
    "training_mode" : "vanilla", # this will be `vanilla` or `pufferfish`, pufferfish will enable the full-rank warmup training and low-rank consecutive training
    "momentum": 0.9,
    "num_epochs": 90,
    "decay_steps" : [30, 60, 80],
    "decay_factor" : 10,
    "warmup_epoch": 5,
    "lr_decay_period": [50,150],
    "lr_decay_factor":0.1,
    "multiplier": 16,
    "lr_warmup_scaling" : 1, # we don not use lr warmup for imagenet task
    "lr_warmup_epochs" : 5, # this is not used
    "optimizer_momentum_type" : "exponential_moving_average",
    "grad_comb":True,
    "early_bird":False,
    "scratch":"./EBTrain-ImageNet/ResNet50/pruned_7008_0.7/pruned.pth.tar",
    "warmup_epochs" : 5  #for learning rate scheduling
}


imagenet_pufferfish_config = {
    "name" : "imagenet",
    #"arch" : "hybrid_resnet50",
    "arch" : "resnet50",
    "lowrank_arch" : "hybrid_resnet50",
    "is_lowrank": True,
    "rank_factor": 4,
    "dataset" : "imagenet",
    "weight_decay": 0.0001,
    "optimizer_weight_decay_conv":0.0001,
    "optimizer_weight_decay_other":0.0001,
    "optimizer_weight_decay_bn":0.0,
    "device" : "cuda:0",
    "data_path": "/home/ubuntu/data",
    "num_dataloader_threads": 8,
    "train_batch_size": 16,
    "test_batch_size": 128,
    "init_lr": 0.1,
    "training_mode" : "pufferfish",
    "momentum": 0.9,
    "num_epochs": 90,
    "full_rank_warmup_epoch" : -1,
    "decay_steps" : [30, 60, 80],
    "decay_factor" : 10,
    "warmup_epoch": 5,
    "lr_decay_period": [50,150],
    "lr_decay_factor":0.1,
    "multiplier": 16,
    "lr_warmup_scaling" : 1, # we don not use lr warmup for imagenet task
    "lr_warmup_epochs" : 5, # this is not used
    "optimizer_momentum_type" : "exponential_moving_average",
    "grad_comb":True,
    "early_bird":False,
    "scratch":"./EBTrain-ImageNet/ResNet50/pruned_7008_0.7/pruned.pth.tar",
    "warmup_epochs" : 5  #for learning rate scheduling
}


cifar10_config_vanilla = {
    "name" : "CNN",
    "arch" : "ResNet18",
    "lowrank_arch" : "LowrankResNet18",
    "dataset" : "Cifar10",
    "device" : "cuda:0",
    "data_path" : "./data/cifar10",
    "num_dataloader_threads" : 1,
    "train_batch_size" : 256,
    "test_batch_size" : 128,
    "full_rank_warmup_epoch" : 80,
    "optimizer_weight_decay_conv":0.0001,
    "optimizer_weight_decay_other":0.0001,
    "optimizer_weight_decay_bn":0.0,
    "init_lr" : 0.1,
    "training_mode" : "vanilla", # this will be `vanilla` or `pufferfish`, pufferfish will enable the full-rank warmup training and low-rank consecutive training
    "optimizer_momentum_type" : "exponential_moving_average",
    "momentum": 0.9,
    "num_epochs": 300,
    "decay_steps": [150, 250],
    "decay_factor" : 0.1, # divide init lr with this
    "switch_freq" : 10,
    "lr_warmup_epochs" : 5,  #for learning rate scheduling
    "lr_warmup_scaling" : 16,
    "grad_comb":True
}


cifar10_config_pufferfish = {
    "name" : "CNN",
    "arch" : "ResNet18",
    "lowrank_arch" : "LowrankResNet18",
    "dataset" : "Cifar10",
    "device" : "cuda:0",
    "data_path" : "./data/cifar10",
    "rank_factor": 4,
    "num_dataloader_threads" : 1,
    "train_batch_size" : 256,
    "test_batch_size" : 128,
    "full_rank_warmup_epoch" : 80,
    "optimizer_weight_decay_conv":0.0001,
    "optimizer_weight_decay_other":0.0001,
    "optimizer_weight_decay_bn":0.0,
    "init_lr" : 0.1,
    "training_mode" : "pufferfish", # this will be `vanilla` or `pufferfish`, pufferfish will enable the full-rank warmup training and low-rank consecutive training
    "optimizer_momentum_type" : "nesterov",
    "momentum": 0.9,
    "num_epochs": 300,
    "decay_steps": [150, 250],
    "decay_factor" : 0.1, # divide init lr with this
    "switch_freq" : 80,
    "lr_warmup_epochs" : 5,  #for learning rate scheduling
    "lowrank_lr_re_warmup_epochs":5,
    "lr_warmup_scaling" : 16,
    "grad_comb":True,
    "optimizer_reducer_rank": 4,
    "optimizer_reducer_reuse_query": True
}


cifar10_config_powerfish = {
    "name" : "CNN",
    "arch" : "ResNet18",
    "lowrank_arch" : "LowrankResNet18",
    "dataset" : "Cifar10",
    "device" : "cuda:0",
    "data_path" : "./data/cifar10",
    "rank_factor": 4,
    "num_dataloader_threads" : 1,
    "train_batch_size" : 256,
    "test_batch_size" : 128,
    "full_rank_warmup_epoch" : 80,
    "optimizer_weight_decay_conv":0.0001,
    "optimizer_weight_decay_other":0.0001,
    "optimizer_weight_decay_bn":0.0,
    "init_lr" : 0.1,
    "training_mode" : "powerfish", # this will be `vanilla` or `pufferfish`, pufferfish will enable the full-rank warmup training and low-rank consecutive training
    "optimizer_momentum_type" : "nesterov",
    "momentum": 0.9,
    "num_epochs": 300,
    "decay_steps": [150, 250],
    "decay_factor" : 0.1, # divide init lr with this
    "switch_freq" : 80,
    "lr_warmup_epochs" : 5,  #for learning rate scheduling
    "lowrank_lr_re_warmup_epochs":5,
    "lr_warmup_scaling" : 16,
    "grad_comb":True,
    "optimizer_reducer_rank": 4,
    "optimizer_reducer_reuse_query": True
}


cifar10_config_vanilla_powersgd = {
    "name" : "CNN",
    "arch" : "ResNet18",
    "lowrank_arch" : "LowrankResNet18",
    "dataset" : "Cifar10",
    "device" : "cuda:0",
    "data_path" : "./data/cifar10",
    "num_dataloader_threads" : 1,
    "train_batch_size" : 256,
    "test_batch_size" : 128,
    "full_rank_warmup_epoch" : 80,
    "optimizer_weight_decay_conv":0.0001,
    "optimizer_weight_decay_other":0.0001,
    "optimizer_weight_decay_bn":0.0,
    "init_lr" : 0.1,
    "training_mode" : "vanilla", # this will be `vanilla` or `pufferfish`, pufferfish will enable the full-rank warmup training and low-rank consecutive training
    "optimizer_momentum_type" : "nesterov",
    "momentum": 0.9,
    "num_epochs": 300,
    "decay_steps": [150, 250],
    "decay_factor" : 0.1, # divide init lr with this
    "switch_freq" : 10,
    "lr_warmup_epochs" : 5,  #for learning rate scheduling
    "lr_warmup_scaling" : 16,
    "grad_comb":True,
    "optimizer_reducer_rank": 2,
    "optimizer_reducer_reuse_query": True
}


cifar10_config_vanilla_signum = {
    "name" : "CNN",
    "arch" : "ResNet18",
    "lowrank_arch" : "LowrankResNet18",
    "dataset" : "Cifar10",
    "device" : "cuda:0",
    "data_path" : "./data/cifar10",
    "num_dataloader_threads" : 1,
    "train_batch_size" : 256,
    "test_batch_size" : 128,
    "full_rank_warmup_epoch" : 80,
    "optimizer_weight_decay_conv":0.0001,
    "optimizer_weight_decay_other":0.0001,
    "optimizer_weight_decay_bn":0.0,
    "init_lr" : 0.00005,
    "training_mode" : "vanilla", # this will be `vanilla` or `pufferfish`, pufferfish will enable the full-rank warmup training and low-rank consecutive training
    "optimizer_momentum_type" : "exponential_moving_average",
    "momentum": 0.9,
    "num_epochs": 300,
    "decay_steps": [150, 250],
    "decay_factor" : 0.1, # divide init lr with this
    "switch_freq" : 10,
    "lr_warmup_epochs" : 5,  #for learning rate scheduling
    "lr_warmup_scaling" : 16,
    "grad_comb":True,
    "optimizer_reducer_reuse_query": True
}



def add_fit_args(parser):
    """
    parser : argparse.ArgumentParser
    return a parser added with args required by fit
    """
    parser.add_argument("--norm-thresh", default=0.2, type=float,
                        help="norm thresh for layer")
    parser.add_argument("--model-type", default="languageModel", type=str,
                        help="type of model helps to select the right config")
    #parser.add_argument("--auto-switch", default=False, action="store_true",
    #                    help="Enables automatic switching")
    # the presence of fixed-k in args will make the value true
    #parser.add_argument("--fixed-k", default=False, action="store_true",
    #                    help="Indicates if we want to use a fixed k")
    #parser.add_argument("--k", default=None, type=int, 
    #                    help= "If fixed-k is true then uses this for training")
    parser.add_argument("--norm-file", type=str, 
                        default="wikitext_lstm_full_rank.json")
    #parser.add_argument("--start-k", default=False, action="store_true",
    #                    help="starts with a k")
    #parser.add_argument("--k-start", default=None, type= int,
    #                    help = "Fix the start k")
    #parser.add_argument("--fixed-sched", default=False, action="store_true",
    #                    help="follow a fixed schedule")
    parser.add_argument("--zero-memory", default=False, action="store_true")
    parser.add_argument("--compressor", type=str, default="vanilla", help="which gradient compressor to use.")
    parser.add_argument("--config-mode", type=str, default="vanilla", help="which framework to use: pufferfish|powerfish|vanilla.") 
    # here powerfish indicates that we conduct powersgd for the full-rank warmup epoch then swtich to full pufferfish

    # distributed arguments
    parser.add_argument("--distributed", default=False, action="store_true",
                        help="Indicates if we have to use distributed")
    parser.add_argument("--master-ip", default=None, type=str,
                        help="Master IP for NCCL/MPI")
    parser.add_argument("--num-nodes", default=0, type=int,
                        help="Indicate number of nodes")
    parser.add_argument("--rank", default=0, type=int,
                        help="Rank of this node")

    args = parser.parse_args()

    return args

def seed(seed):
    # seed = 1234
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    #TODO: Do we need deterministic in cudnn ? Double check
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print ("Seeded everything")


def norm_calculator(model):
    model_norm = 0
    for param_index, param in enumerate(model.parameters()):
        model_norm += torch.norm(param) ** 2
    return torch.sqrt(model_norm).item()


def get_lr(config, epoch_num):
    """
    Return learning rate in case of the time 
    """
    max_factor = config['lr_warmup_scaling']
    factor = 1.0 + (max_factor - 1.0) *min(epoch_num/config['lr_warmup_epochs'], 1.0)
    if config['name'] == "CNN" or config['name'] == 'cifar100' or config['name'] == 'svhn':
        if epoch_num < 150:
            if epoch_num in range(config["lr_warmup_epochs"]):
                new_lr = config['init_lr'] * factor
            else:
                # sometimes pufferfish+powersgd suffers from large batch training after full rank warmup
                # so let's do one simple trick here that after full-rank warmup, we do another lr warmup
                if config['training_mode'] == "pufferfish":
                    if epoch_num in range(config["full_rank_warmup_epoch"], config["full_rank_warmup_epoch"]+config['lowrank_lr_re_warmup_epochs']):
                        factor = 1.0 + (max_factor - 1.0) *min((epoch_num - config["full_rank_warmup_epoch"])/config["lowrank_lr_re_warmup_epochs"], 1.0)
                        new_lr = config['init_lr'] * factor
                    else:
                        new_lr = config['init_lr'] * max_factor
                else:
                    new_lr = config['init_lr'] * max_factor
            return new_lr
        elif epoch_num >= 150 and epoch_num <250:
            new_lr = config['init_lr'] * max_factor/10.0
            return new_lr
        elif epoch_num >= 250:
            new_lr = config['init_lr'] * max_factor/100.0
            return new_lr
        else:
            print ("Something went wrong in learning rate selection")
    if config['name'] == 'imagenet':
        if epoch_num in range(30):
            new_lr = config['init_lr']
        elif epoch_num in range(30, 60):
            new_lr = config['init_lr']/10.0
        elif epoch_num in range(60, 80):
            new_lr = config['init_lr']/100.0
        elif epoch_num in range(80, 90):
            new_lr = config['init_lr']/1000.0
        else:
            raise NotImplementedError("Invalid Epoch ....")
    return new_lr


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


def vectorize_grad(grad_train):
    #return torch.cat([grad_value.view(-1) for grad_value in grad_train])
    # better implementation?
    return _flatten_dense_tensors(grad_train)

def devectorize_grad(reduced_grad, model):
    out_grad_list = []
    index_bias = 0
    for p_index, p in enumerate(model.parameters()):
        out_grad_list.append(reduced_grad[index_bias:index_bias+p.numel()].view(p.size()))
        index_bias += p.numel()
    return out_grad_list


def replace_grad_by_momentum(grad, momentum, config=None):
    """
    Inplace operation that applies momentum to a gradient.
    This distinguishes between types of momentum (heavy-ball vs nesterov)
    """
    if config["optimizer_momentum_type"] == "heavy-ball":
        grad[:] = momentum
    if config["optimizer_momentum_type"] == "exponential_moving_average":
        grad[:] = momentum
    elif config["optimizer_momentum_type"] == "nesterov":
        grad[:] += momentum
    else:
        raise ValueError("Unknown momentum type")


def main(args):
    chosen_method_log = dict() # this writes things when method is changed
    current_method_log = dict() # this will monitor what is the current method 
    candidate_method_stat = dict() # this tracks the thresh for all candidate method
    timing_log = defaultdict(list)
    floats_communicated = dict()
    #grad_calc_dict = dict()
    ratio_calc_dict = dict()
    compute_time_per_dict = defaultdict(dict)
    breakdown_time_log_dict = dict()

    prev_norm = None
    json_f_name =os.path.basename(args.norm_file).split('.')[0] + '.json'
    current_method_log_fname = os.path.basename(
        args.norm_file).split('.')[0] + "_per_epoch_method.json"
    candidate_methods_stat_fname = os.path.basename(
        args.norm_file).split('.')[0] + "_candidate_method_stats.json"
    timing_log_fname = os.path.basename(
        args.norm_file).split('.')[0] + "_timing_log.json"
    bytes_log_fname = os.path.basename(
        args.norm_file).split('.')[0] + "_floats_communicated.json"
    ratio_log_fname = os.path.basename(
        args.norm_file).split('.')[0] + "_ratio_vals.json"
    grad_calc_fname = os.path.basename(
        args.norm_file).split('.')[0] + "_grad_norm_vals.json"
    per_iteration_compute_time_log = os.path.basename(
        args.norm_file).split('.')[0] + "_per_iteration_compute_time.json"
    breakdown_time_log_fname = os.path.basename(
        args.norm_file).split('.')[0] + "_per_epoch_breakdown_time.json"

    #TODO: Clean this up to manually select the model 
    if args.model_type == "CNN":
        if args.config_mode == "vanilla":
            if args.compressor == "powersgd":
                config = cifar10_config_vanilla_powersgd
            elif args.compressor == "signum":
                config = cifar10_config_vanilla_signum
            else:
                config = cifar10_config_vanilla
        elif args.config_mode == "pufferfish":
            # TODO: let's add pufferfish + powersgd here
            config = cifar10_config_pufferfish
        elif args.config_mode == "powerfish":
            config = cifar10_config_powerfish
    elif args.model_type == "languageModel":
        config = lstm_config
    elif args.model_type == "newlanguageModel":
        config = new_lstm_config
    elif args.model_type == "imagenet":
        #config = imagenet_config
        if args.config_mode == "vanilla":
            config = imagenet_vanilla_config
        elif args.config_mode == "pufferfish":
            config = imagenet_pufferfish_config
        else:
            raise NotImplementedError("unsupported config mode ...")
    elif args.model_type == "cifar10":
        config = cifar10_config
    elif args.model_type == "svhn":
        config = svhn_config
    elif args.model_type == "squeezenet_cifar":
        config = cifar_squeezenet_config
    else:
        raise NotImplemented("{} not NotImplemented".format(args.model_type))
    config['is_distributed'] = False # adding a new key in the config
    if args.distributed:
        print ("Initializing distributed")
        dist.init_process_group(backend="NCCL", init_method=args.master_ip,
                                timeout=datetime.timedelta(seconds=120),
                                world_size=args.num_nodes, rank=args.rank)
        config['is_distributed'] = True 
        print ("Distributed Initialized")
    train_task = train_network.build(config['dataset'], config)
    logger.info("==> Model Architecture: {}".format(train_task.model))

    if config['training_mode'] == "pufferfish":
        logger.info("==> Lowrank Model Architecture: {}".format(train_task.lowrank_model))

    #TODO: Fix this for distributed
    # use parameter groups to get things for different learning rates
    # and weight decay parameters 
    current_lr = config['init_lr']

    print("Initi model Norm: {}, rank: {}".format(norm_calculator(train_task.model), args.rank))

    if args.compressor not in ("signum", "powersgd"):
        if config['name'] == "CNN" or config['name'] == 'cifar100' or config['name'] == 'svhn':
            # optimizer only for langauge model
            # otherwise we are going manual\
            # my guess is that repackage thing for language models changes
            # the model structure and the optimizer is registered only for some of
            # the parameters
            #optimizer = optim.SGD(train_task.model.parameters(), lr=current_lr,
            #                                momentum=config['momentum'],
            #                                weight_decay=1e-4)
            parameters = add_weight_decay(train_task.model, weight_decay=1e-4)
            optimizer = optim.SGD(parameters, lr=current_lr,
                                           momentum=config['momentum'],
                                           weight_decay=0)


            if config['training_mode'] == "pufferfish":
                optimizer_lowrank = optim.SGD(train_task.lowrank_model.parameters(), 
                                                    lr=current_lr*config['lr_warmup_scaling'],
                                                    momentum=config['momentum'],
                                                    weight_decay=1e-4)            

            if config['training_mode'] == "vanilla":
                scheduler_multi_step = torch.optim.lr_scheduler.MultiStepLR(optimizer, 
                                                            milestones=[e-config["lr_warmup_epochs"]-1 for e in config["decay_steps"]], 
                                                            gamma=config["decay_factor"])
                scheduler_warmup = GradualWarmupScheduler(optimizer, 
                                multiplier=config["lr_warmup_scaling"], 
                                total_epoch=config["lr_warmup_epochs"], 
                                after_scheduler=scheduler_multi_step)
            elif config['training_mode'] == "pufferfish":
                scheduler_multi_step = torch.optim.lr_scheduler.MultiStepLR(optimizer, 
                                                            milestones=[e-config["lr_warmup_epochs"]-1 for e in config["decay_steps"]], 
                                                            gamma=config["decay_factor"])
                scheduler_warmup = GradualWarmupScheduler(optimizer, 
                                multiplier=config["lr_warmup_scaling"], 
                                total_epoch=config["lr_warmup_epochs"], 
                                after_scheduler=scheduler_multi_step)
                scheduler_multi_step_lowrank = torch.optim.lr_scheduler.MultiStepLR(optimizer_lowrank, 
                                                            milestones=[e-config["full_rank_warmup_epoch"]-1 for e in config["decay_steps"]], 
                                                            gamma=config["decay_factor"])                               
            else:
                raise NotImplementedError("Unsupported training mode !!!")               
        if config['name'] == 'imagenet':
            # parameters 
            parameters = add_weight_decay(train_task.model, config['weight_decay'])
            # weight decay is incorporated in the parameters
            optimizer = optim.SGD(parameters, lr=current_lr,
                                  momentum=config['momentum'], weight_decay=0)

            if config['training_mode'] == "pufferfish":
                lowrank_parameters = add_weight_decay(train_task.lowrank_model, config['weight_decay'])
                optimizer_lowrank = optim.SGD(lowrank_parameters, 
                                                    lr=current_lr,
                                                    momentum=config['momentum'],
                                                    weight_decay=0) 

            scheduler_multi_step = lr_scheduler.MultiStepLR(optimizer, 
                                                            milestones=[e for e in config['decay_steps']], 
                                                            gamma=config['lr_decay_factor'])

        if config['name'] == "squeezenet_cifar":
            # special optimizer for squeezenet
            optimizer = optim.SGD(train_task.model.parameters(), lr=current_lr,
                                  momentum=config['momentum'],
                                  weight_decay=5e-4)
    else:
        optimizer = None

    
    current_test_loss = None
    best_test_loss = None

    if args.compressor in ("signum", "powersgd"):
        momenta = [torch.empty_like(param) for param in train_task.model.parameters()]
        if config['training_mode'] == "pufferfish":
            momenta_lowrank = [torch.empty_like(param) for param in train_task.lowrank_model.parameters()]
    if args.compressor == "powersgd":
        # handle error feedback
        memories = [torch.zeros_like(param) for param in train_task.model.parameters()]
        send_buffers = [torch.zeros_like(param) for param in train_task.model.parameters()]
        if config['training_mode'] == "pufferfish":
            # handle error feedback lowrank
            memories_lowrank = [torch.zeros_like(param) for param in train_task.lowrank_model.parameters()]
            send_buffers_lowrank = [torch.zeros_like(param) for param in train_task.lowrank_model.parameters()]
            # for powerfish, we don't need to initialize these buffers as we switches to normal low-rank training         

    if args.compressor == "vanilla":
        grad_compressor = None
    elif args.compressor == "suquantization":
        grad_compressor = StochasticUniformQuantization(random_seed=0, device=config['device'])
    elif args.compressor == "signum":
        grad_compressor = SignSGDwithMajorityVoteReducer(random_seed=0, device=config['device'])
    elif args.compressor == "powersgd":
        grad_compressor = RankKReducer(random_seed=0, device=config['device'], 
                                        n_power_iterations=0, 
                                        reuse_query=config['optimizer_reducer_reuse_query'], 
                                        rank=config['optimizer_reducer_rank'])
        if config['training_mode'] == "pufferfish":
            grad_compressor_lowrank = RankKReducer(random_seed=0, device=config['device'], 
                                            n_power_iterations=0, 
                                            reuse_query=config['optimizer_reducer_reuse_query'], 
                                            rank=config['optimizer_reducer_rank'])
            # for powerfish, we don't need to initialize these buffers as we switches to normal low-rank training         
    else:
        raise NotImplementedError("Unsupported gradient compressor !")

    
    wds = [get_weight_decay(name, config) for name in train_task.parameter_names]

    if config['training_mode'] == "pufferfish":
        lowrank_wds = [get_weight_decay(name, config) for name in train_task.lowrank_parameter_names]
        # for powerfish, we don't need to initialize these buffers as we switches to normal low-rank training

    for epoch in range(config['num_epochs']):
        # to put into the `breakdown_time_log`
        epoch_compute_time = 0.0
        epoch_comm_time = 0.0
        epoch_total_time = 0.0
        epoch_encoding_overhead = 0.0
        epoch_iter_time = 0.0

        # for logging out the current learning rate
        if args.compressor not in ("signum", "powersgd"):
            if config['training_mode'] == "pufferfish":
                if epoch in range(config['full_rank_warmup_epoch']):
                    for param_group in optimizer.param_groups:
                        logger.info("### Epoch: {}, Current Effective lr: {}".format(epoch, param_group['lr']))
                        break
                else:
                    for param_group in optimizer_lowrank.param_groups:
                        logger.info("### Epoch: {}, Current Effective lr: {}".format(epoch, param_group['lr']))
                        break              
            elif config['training_mode'] == "vanilla":
                for param_group in optimizer.param_groups:
                    logger.info("### Epoch: {}, Current Effective lr: {}".format(epoch, param_group['lr']))
                    break
            else:
                raise NotImplementedError("Unsupported training mode !!!")
        else:
            if config['training_mode'] == "powerfish":
                if epoch in range(config['full_rank_warmup_epoch']):
                    current_lr = get_lr(config=config, epoch_num=epoch)
                    logger.info("### Epoch: {}, Current Effective lr: {}".format(epoch, current_lr))
                elif epoch == config['full_rank_warmup_epoch']:
                    pass
                else:
                    for param_group in optimizer_lowrank.param_groups:
                        logger.info("### Epoch: {}, Current Effective lr: {}".format(epoch, param_group['lr']))
                        break  
            else:
                current_lr = get_lr(config=config, epoch_num=epoch)
                logger.info("### Epoch: {}, Current Effective lr: {}".format(epoch, current_lr))

        if config['is_distributed']:
            train_task.sampler.set_epoch(epoch) # set epoch to make sure the data is reshuffled per epoch

        elements_per_epoch = 0

        if config['training_mode'] in ("pufferfish", "powerfish") and epoch == config["full_rank_warmup_epoch"]:
            train_task.init_hybrid_net()

            if args.compressor not in ("signum", "powersgd") or config['training_mode'] == "powerfish":
                # let's reset the optimizer here:
                lowrank_parameters = add_weight_decay(train_task.lowrank_model, weight_decay=1e-4)
                
                if config['training_mode'] == "pufferfish":
                    optimizer_lowrank = optim.SGD(lowrank_parameters, 
                                            lr=config['init_lr']*config['lr_warmup_scaling'],
                                            momentum=config['momentum'],
                                            weight_decay=0)
                    scheduler_multi_step_lowrank = torch.optim.lr_scheduler.MultiStepLR(optimizer_lowrank, 
                                                                milestones=[e-config["full_rank_warmup_epoch"] for e in config["decay_steps"]], 
                                                                gamma=config["decay_factor"])              
                elif config['training_mode'] == "powerfish":
                    optimizer_lowrank = optim.SGD(lowrank_parameters, 
                                            lr=config['init_lr']*config['lr_warmup_scaling'],
                                            momentum=config['momentum'],
                                            weight_decay=0)
                    scheduler_multi_step_lowrank = torch.optim.lr_scheduler.MultiStepLR(optimizer_lowrank, 
                                                                milestones=[e-config["full_rank_warmup_epoch"] for e in config["decay_steps"]], 
                                                                gamma=config["decay_factor"])  
                    #scheduler_multi_step_lowrank = torch.optim.lr_scheduler.MultiStepLR(optimizer_lowrank, 
                    #                                            milestones=[e-config["full_rank_warmup_epoch"]-config["lowrank_lr_re_warmup_epochs"]-1 for e in config["decay_steps"]], 
                    #                                            gamma=config["decay_factor"])
                    #scheduler_warmup_powerfish = GradualWarmupScheduler(optimizer_lowrank, 
                    #                    multiplier=config["lr_warmup_scaling"], 
                    #                    total_epoch=config["lowrank_lr_re_warmup_epochs"], 
                    #                    after_scheduler=scheduler_multi_step_lowrank)
                    for param_group in optimizer_lowrank.param_groups:
                        logger.info("### Epoch: {}, Current Effective lr: {}".format(epoch, param_group['lr']))
                        break
                else:
                    raise NotImplementedError("Unsupported training mode ...")

                optimizer_lowrank.zero_grad()
            else:
                optimizer_lowrank = None
                scheduler_multi_step_lowrank = None
                train_task.lowrank_model.zero_grad()


        # note that currently we assume everything is running over CUDA
        if config['training_mode'] in ("pufferfish", "powerfish"):
            if epoch in range(config['full_rank_warmup_epoch']):
                print("epoch: {}, operating vanilla model ....".format(epoch))
                train_task.model.train()
            else:
                print("epoch: {}, operating low rank model ....".format(epoch))
                train_task.lowrank_model.train()
        elif config['training_mode'] == "vanilla":
            train_task.model.train()
        else:
            raise NotImplementedError("Unsupported training mode !!!")


        for iter_index, (data, target) in enumerate(train_task.train_loader):
            comm_start = torch.cuda.Event(enable_timing=True)
            comm_end = torch.cuda.Event(enable_timing=True)
            comp_start = torch.cuda.Event(enable_timing=True)
            comp_end = torch.cuda.Event(enable_timing=True)
            iter_start = torch.cuda.Event(enable_timing=True)
            iter_end = torch.cuda.Event(enable_timing=True)

            debug_start = torch.cuda.Event(enable_timing=True)
            debug_end = torch.cuda.Event(enable_timing=True)

            iter_start.record()

            out_grad_list = list() #list to store output gradients

            comp_start.record()
            grad_train = train_task.batch_loss_and_gradient(batch_idx=iter_index, data=data, target=target, logger=logger, epoch=epoch)
            comp_end.record()
            torch.cuda.synchronize()
            iter_comp_dur = float(comp_start.elapsed_time(comp_end))/1000.0
            epoch_compute_time += iter_comp_dur
            #print("@@@@@@ Epoch: {} Iter: {} Iter Comp Dur: {}".format(epoch, iter_index, iter_comp_dur))

            if args.compressor == "signum":
                # based on the discussion in https://arxiv.org/pdf/1810.05291.pdf, 
                # momentum rather than gradient is compressed
                # we thus calculate the momentum first
                for grad, momentum in zip(grad_train, momenta):
                    if epoch == 0 and iter_index == 0:
                        momentum.data = grad.clone().detach()
                    else:
                        momentum.mul_(config["momentum"]).add_(
                                                                alpha=1 - config["momentum"], other=grad
                                                            )
                        replace_grad_by_momentum(grad, momentum, config=config)

            # aggregate the gradients here:
            if args.compressor != "powersgd":
                concat_grad = vectorize_grad(grad_train)
            else:
                # implement err feedback step here
                if config['training_mode'] == "pufferfish":
                    if epoch in range(config['full_rank_warmup_epoch']):
                        for grad, memory, send_bfr in zip(grad_train, memories, send_buffers):
                            send_bfr.data[:] = grad + memory
                    else:
                        for grad, memory, send_bfr in zip(grad_train, memories_lowrank, send_buffers_lowrank):
                            send_bfr.data[:] = grad + memory
                elif config['training_mode'] == "vanilla":
                    for grad, memory, send_bfr in zip(grad_train, memories, send_buffers):
                        send_bfr.data[:] = grad + memory
                elif config['training_mode'] == "powerfish":
                    if epoch in range(config['full_rank_warmup_epoch']):
                        for grad, memory, send_bfr in zip(grad_train, memories, send_buffers):
                            send_bfr.data[:] = grad + memory
                    else:
                        # we switch back to normal low-rank training
                        concat_grad = vectorize_grad(grad_train)
                else:
                    raise NotImplementedError("Unsupported training mode !!!")

            # communication step
            if args.compressor == "vanilla":
                comm_start.record()
            elif config['training_mode'] == "powerfish":
                if epoch not in range(config['full_rank_warmup_epoch']):
                    comm_start.record()

            if config['grad_comb']:
                if args.compressor == "vanilla":
                    torch.distributed.all_reduce(concat_grad, async_op=False)
                    concat_grad[:] = concat_grad/args.num_nodes
                elif args.compressor == "suquantization":
                    #print("##### max: {}, min grad: {}".format(torch.max(concat_grad), torch.min(concat_grad)))
                    reduced_aggregated_grad, bits_communicated, compressor_iter_comm_time, iter_encode_decode_time = grad_compressor.reduce(concat_grad)
                    concat_grad[:] = reduced_aggregated_grad/args.num_nodes
                elif args.compressor == "signum":
                    reduced_aggregated_grad, bits_communicated, compressor_iter_comm_time, iter_encode_decode_time = grad_compressor.reduce(concat_grad)
                    concat_grad[:] = reduced_aggregated_grad
                elif args.compressor == "powersgd":
                    if config['training_mode'] == "pufferfish":
                        if epoch in range(config['full_rank_warmup_epoch']):
                            bits_communicated, compressor_iter_comm_time, iter_encode_decode_time = grad_compressor.reduce(grad_in=send_buffers, 
                                                                    grad_out=grad_train, 
                                                                    memory_out=memories)
                        else:
                            bits_communicated, compressor_iter_comm_time, iter_encode_decode_time = grad_compressor_lowrank.reduce(grad_in=send_buffers_lowrank, 
                                                                    grad_out=grad_train, 
                                                                    memory_out=memories_lowrank)
                        print("@@@@@@ Epoch: {}, iter: {}, comm_time: {}".format(epoch, iter_index, compressor_iter_comm_time))
                    elif config['training_mode'] == "powerfish":
                        if epoch in range(config['full_rank_warmup_epoch']):
                            bits_communicated, compressor_iter_comm_time, iter_encode_decode_time = grad_compressor.reduce(grad_in=send_buffers, 
                                                                    grad_out=grad_train, 
                                                                    memory_out=memories)
                            print("@@@@@@ Epoch: {}, iter: {}, comm_time: {}".format(epoch, iter_index, compressor_iter_comm_time))
                        else:
                            torch.distributed.all_reduce(concat_grad, async_op=False)
                            concat_grad[:] = concat_grad/args.num_nodes
                    elif config['training_mode'] == "vanilla":
                        bits_communicated, compressor_iter_comm_time, iter_encode_decode_time = grad_compressor.reduce(grad_in=send_buffers, 
                                                                    grad_out=grad_train, 
                                                                    memory_out=memories)
                    else:
                        raise NotImplementedError("Unsupported training mode !!!")
                else:
                    raise NotImplementedError("Unsupported gradient compressor !")

            if args.compressor == "vanilla":
                comm_end.record()
                torch.cuda.synchronize()
            elif config['training_mode'] == "powerfish":
                if epoch not in range(config['full_rank_warmup_epoch']):
                    comm_end.record()
                    torch.cuda.synchronize()                    

            if config['grad_comb']:
                if config['training_mode'] == "pufferfish":
                    if args.compressor != "powersgd":
                        if epoch in range(config['full_rank_warmup_epoch']):
                            out_grad_list = devectorize_grad(concat_grad, train_task.model)
                        else:
                            out_grad_list = devectorize_grad(concat_grad, train_task.lowrank_model)
                    else:
                        out_grad_list = [g for g in grad_train]
                elif config['training_mode'] == "vanilla":
                    if args.compressor != "powersgd":
                        out_grad_list = devectorize_grad(concat_grad, train_task.model)
                    else:
                        out_grad_list = [g for g in grad_train]
                elif config['training_mode'] == "powerfish":
                    if epoch in range(config['full_rank_warmup_epoch']):
                        out_grad_list = [g for g in grad_train]
                    else:
                        out_grad_list = devectorize_grad(concat_grad, train_task.lowrank_model)
                else:
                    raise NotImplementedError("Unsupported training mode !!!")

                if args.compressor == "vanilla":
                    iter_comm_cost = float(comm_start.elapsed_time(comm_end))/1000.0
                    epoch_comm_time += iter_comm_cost
                elif args.compressor in ("suquantization", "signum", "powersgd"):
                    if config['training_mode'] == "powerfish":
                        if epoch in range(config['full_rank_warmup_epoch']):
                            epoch_comm_time += compressor_iter_comm_time
                            epoch_encoding_overhead += iter_encode_decode_time                            
                        else:
                            iter_comm_cost = float(comm_start.elapsed_time(comm_end))/1000.0
                            epoch_comm_time += iter_comm_cost                            
                    else:
                        epoch_comm_time += compressor_iter_comm_time
                        epoch_encoding_overhead += iter_encode_decode_time
                else:
                    raise NotImplementedError("Unsupported gradient compressor !")
            
            # updated the gradients in place
            # TODO: Move this to a new function
            if args.compressor not in ("signum", "powersgd"):
                if config['training_mode'] == "pufferfish":
                    if epoch in range(config['full_rank_warmup_epoch']):
                        for idx, param in enumerate(train_task.model.parameters()):
                            param.grad.data = out_grad_list[idx]
                    else:
                        for idx, param in enumerate(train_task.lowrank_model.parameters()):
                            param.grad.data = out_grad_list[idx]
                elif config['training_mode'] == "vanilla":
                        for idx, param in enumerate(train_task.model.parameters()):
                            param.grad.data = out_grad_list[idx]
                else:
                    raise NotImplementedError("Unsupported training mode !!!")


                if config['name'] == 'CNN' or config['name'] == 'cifar100' or config['name'] == 'imagenet' or config['name'] == 'svhn':
                    if config['training_mode'] == "pufferfish":
                        if epoch in range(config['full_rank_warmup_epoch']):
                            optimizer.step()
                            optimizer.zero_grad()
                        else:
                            optimizer_lowrank.step()
                            optimizer_lowrank.zero_grad()
                    elif config['training_mode'] == "vanilla":
                        optimizer.step()
                        optimizer.zero_grad()
                    else:
                        raise NotImplementedError("Unsupported training mode !!!")

                elif config['name'] == "squeezenet_cifar":
                    optimizer.step()
                    optimizer.zero_grad()
                elif config['name'] == 'languageModel' or config['name'] == 'newlanguageModel':
                    # momentum implementation 
                    for idx, param in enumerate(train_task.model.parameters()):
                        if epoch == 0 and first_iter == 0:
                            momenta[idx].data = param.grad.data.clone().detach()
                            first_iter = 1
                        else:
                            momenta[idx].data.mul_(0.9).add_(param.grad.data)
                        param.grad.data[:] += momenta[idx].data

                    for p in train_task.model.parameters():
                        p.data.add_(-current_lr, p.grad.data)
                    train_task.model.zero_grad()
                else:
                    raise NotImplementedError("Unsupported model name type ...")
            else:
                # for signsgd we will need to handle weight decay manually: (line ``update parameters"" in https://openreview.net/pdf?id=BJxhijAcY7)
                if config['training_mode'] == "pufferfish":
                    if epoch in range(config['full_rank_warmup_epoch']):
                        for grad, param, wd in zip(out_grad_list, train_task.model.parameters(), wds):
                            if wd > 0:
                                grad.add_(alpha=wd, other=param.data.detach())
                    else:
                        for grad, param, wd in zip(out_grad_list, train_task.lowrank_model.parameters(), lowrank_wds):
                            if wd > 0:
                                grad.add_(alpha=wd, other=param.data.detach())
                elif config['training_mode'] == "vanilla":
                    for grad, param, wd in zip(out_grad_list, train_task.model.parameters(), wds):
                        if wd > 0:
                            grad.add_(alpha=wd, other=param.data.detach())
                elif config['training_mode'] == "powerfish":
                    if epoch in range(config['full_rank_warmup_epoch']):
                        for grad, param, wd in zip(out_grad_list, train_task.model.parameters(), wds):
                            if wd > 0:
                                grad.add_(alpha=wd, other=param.data.detach())
                    else:
                        for idx, param in enumerate(train_task.lowrank_model.parameters()):
                            param.grad.data = out_grad_list[idx]                        
                else:
                    raise NotImplementedError("Unsupported training mode !!!")

                if args.compressor == "powersgd":
                    if config['training_mode'] == "pufferfish":
                        if epoch in range(config['full_rank_warmup_epoch']):
                            for grad, momentum in zip(out_grad_list, momenta):
                                if epoch == 0 and iter_index == 0:
                                    momentum.data = grad.clone().detach()
                                else:
                                    if (
                                        config["optimizer_momentum_type"]
                                        == "exponential_moving_average"
                                    ):
                                        momentum.mul_(config["momentum"]).add_(
                                            alpha=(1 - config["momentum"]), other=grad
                                        )
                                    else:
                                        momentum.mul_(config["momentum"]).add_(other=grad)
                                replace_grad_by_momentum(grad, momentum, config=config)
                        else:
                            for grad, momentum in zip(out_grad_list, momenta_lowrank):
                                if epoch == 0 and iter_index == 0:
                                    momentum.data = grad.clone().detach()
                                else:
                                    if (
                                        config["optimizer_momentum_type"]
                                        == "exponential_moving_average"
                                    ):
                                        momentum.mul_(config["momentum"]).add_(
                                            alpha=(1 - config["momentum"]), other=grad
                                        )
                                    else:
                                        momentum.mul_(config["momentum"]).add_(other=grad)
                                replace_grad_by_momentum(grad, momentum, config=config)
                    elif config['training_mode'] == "vanilla":
                        for grad, momentum in zip(out_grad_list, momenta):
                            if epoch == 0 and iter_index == 0:
                                momentum.data = grad.clone().detach()
                            else:
                                if (
                                    config["optimizer_momentum_type"]
                                    == "exponential_moving_average"
                                ):
                                    momentum.mul_(config["momentum"]).add_(
                                        alpha=(1 - config["momentum"]), other=grad
                                    )
                                else:
                                    momentum.mul_(config["momentum"]).add_(other=grad)
                            replace_grad_by_momentum(grad, momentum, config=config)
                    elif config['training_mode'] == "powerfish":
                        if epoch in range(config['full_rank_warmup_epoch']):
                            for grad, momentum in zip(out_grad_list, momenta):
                                if epoch == 0 and iter_index == 0:
                                    momentum.data = grad.clone().detach()
                                else:
                                    if (
                                        config["optimizer_momentum_type"]
                                        == "exponential_moving_average"
                                    ):
                                        momentum.mul_(config["momentum"]).add_(
                                            alpha=(1 - config["momentum"]), other=grad
                                        )
                                    else:
                                        momentum.mul_(config["momentum"]).add_(other=grad)
                                replace_grad_by_momentum(grad, momentum, config=config)
                        else:
                            pass
                    else:
                        raise NotImplementedError("Unsupported training mode !!!")       

                if config['training_mode'] == "pufferfish":
                    if epoch in range(config['full_rank_warmup_epoch']):
                        for grad, p in zip(out_grad_list, train_task.model.parameters()):
                            p.data.add_(alpha=-current_lr, other=grad)
                        train_task.model.zero_grad()
                    else:
                        for grad, p in zip(out_grad_list, train_task.lowrank_model.parameters()):
                            p.data.add_(alpha=-current_lr, other=grad)
                        train_task.lowrank_model.zero_grad()
                elif config['training_mode'] == "vanilla":
                    for grad, p in zip(out_grad_list, train_task.model.parameters()):
                        p.data.add_(alpha=-current_lr, other=grad)
                    train_task.model.zero_grad()
                elif config['training_mode'] == "powerfish":
                    if epoch in range(config['full_rank_warmup_epoch']):
                        for grad, p in zip(out_grad_list, train_task.model.parameters()):
                            p.data.add_(alpha=-current_lr, other=grad)
                        train_task.model.zero_grad()
                    else:
                        optimizer_lowrank.step()
                        optimizer_lowrank.zero_grad()                                                
                else:
                    raise NotImplementedError("Unsupported training mode !!!") 

                #train_task.model.zero_grad()

            iter_end.record()
            torch.cuda.synchronize()
            iter_total_dur = float(iter_start.elapsed_time(iter_end))/1000.0
            epoch_iter_time += iter_total_dur
            # print("@@@@@@@ Iter: {}, Comp: {}, Comm: {}, Total: {}, Debug Dur: {}".format(iter_index, 
            #                                                     iter_comp_dur,
            #                                                     iter_comm_cost,
            #                                                     iter_total_dur, debug_dur))

        breakdown_time_log_dict[epoch] = {"comp":epoch_compute_time, "comm":epoch_comm_time, 
                                        "encdec_overhead":epoch_encoding_overhead, "total":epoch_iter_time}

        floats_communicated[epoch] = elements_per_epoch

        torch.distributed.barrier()        

        with open(bytes_log_fname, "w") as fout:
            json.dump(floats_communicated, fout)
        with open(per_iteration_compute_time_log, "w") as fout:
            json.dump(compute_time_per_dict, fout)
        with open(breakdown_time_log_fname, "w") as fout:
            json.dump(breakdown_time_log_dict, fout)

        # validate model
        if config['training_mode'] in ("pufferfish", "powerfish"):
            if epoch in range(config['full_rank_warmup_epoch']):
                current_test_loss = train_task.validate_model(logger)
            else:
                current_test_loss = train_task.validate_lowrank_model(logger)
        elif config['training_mode'] == "vanilla":
            current_test_loss = train_task.validate_model(logger)
        else:
            raise NotImplementedError("Unsupported training mode !!!")

        current_test_loss = 10000
        if not best_test_loss or current_test_loss < best_test_loss:
            best_test_loss = current_test_loss

        if args.compressor not in ("signum", "powersgd"):
            if config['name'] == 'CNN' or config['name'] == 'cifar100' or config['name'] == 'svhn':
                if config['training_mode'] == "pufferfish":
                    if epoch in range(config['full_rank_warmup_epoch']):
                        scheduler_warmup.step()
                    else:
                        scheduler_multi_step_lowrank.step()
                elif config['training_mode'] == "vanilla":
                    scheduler_warmup.step()
                else:
                    raise NotImplementedError("Unsupported training mode !!!")
            if config['name'] == 'imagenet':
                scheduler_multi_step.step()
        else:
            if config['name'] == 'CNN' or config['name'] == 'cifar100' or config['name'] == 'svhn':
                if config['training_mode'] == "powerfish":
                    if epoch in range(config['full_rank_warmup_epoch']):
                        pass
                    else:
                        scheduler_multi_step_lowrank.step()
                        #scheduler_warmup_powerfish.step()

if __name__ == "__main__":
    # making sure seed is the first thing to be called
    seed(42) # 43 44
    args = add_fit_args(argparse.ArgumentParser(description='Auto Scale'))
    log_file_name = os.path.basename(args.norm_file).split(".")[0] + ".log"
    logging.basicConfig(filename=log_file_name)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.info("Arguments: {}".format(args))
    print(args)
    #main(dataset="Cifar10", jl=True)
    main(args)
