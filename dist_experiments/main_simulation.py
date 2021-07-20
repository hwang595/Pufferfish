# in this script, we will simulate N workers on each GPU
# the way to simulate is to calculate gradient for N times, and then conduct communication

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
from gradient_reducers import StochasticUniformQuantization, SignSGDwithMajorityVoteReducerSimulation

from torch.optim import lr_scheduler
from warmup_scheduler import GradualWarmupScheduler


from utils import *

# added files
# import grad_utils
import train_network
import sparsify_gradient
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors

# commit to say powersgd wrap
auto_scale_high = 2
auto_scale_low = 1


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
    "momentum": 0.9,
    "num_epochs": 90,
    "decay_steps" : [30, 60, 80],
    "decay_factor" : 10,
    "warmup_epoch": 5,
    "lr_decay_period": [50,150],
    "lr_decay_factor":0.1,
    "multiplier": 16,
    "switch_freq":10,
    "grad_comb":True,
    "early_bird":False,
    "scratch":"./EBTrain-ImageNet/ResNet50/pruned_7008_0.7/pruned.pth.tar",
    "warmup_epochs" : 5  #for learning rate scheduling
}


imagenet_pufferfish_config = {
    "name" : "imagenet",
    "arch" : "hybrid_resnet50",
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
    "momentum": 0.9,
    "num_epochs": 90,
    "decay_steps" : [30, 60, 80],
    "decay_factor" : 10,
    "warmup_epoch": 5,
    "lr_decay_period": [50,150],
    "lr_decay_factor":0.1,
    "multiplier": 16,
    "switch_freq":10,
    "grad_comb":True,
    "early_bird":False,
    "scratch":"./EBTrain-ImageNet/ResNet50/pruned_7008_0.7/pruned.pth.tar",
    "warmup_epochs" : 5  #for learning rate scheduling
}


cifar10_config = {
    "name" : "CNN",
    "arch" : "ResNet18",
    "dataset" : "Cifar10",
    "device" : "cuda:0",
    "data_path" : "./data/cifar10",
    "num_dataloader_threads" : 1,
    "train_batch_size" : 128,
    "test_batch_size" : 128,
    "optimizer_weight_decay_conv":0.0001,
    "optimizer_weight_decay_other":0.0001,
    "optimizer_weight_decay_bn":0.0,
    "init_lr" : 0.0002,
    "momentum": 0.9,
    "num_epochs": 300,
    "decay_steps": [150, 250],
    "decay_factor" : [10, 10], # divide init lr with this
    "switch_freq" : 10,
    "warmup_epochs" : 5,  #for learning rate scheduling
    "grad_comb":True
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
    parser.add_argument("--num-simulated-nodes", default=1, type=int, 
                        help= "number of nodes to simulate.")
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
    parser.add_argument("--config-mode", type=str, default="vanilla", help="which framework to use: pufferfish|vanilla.")

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

def get_lr(config, epoch_num):
    """
    Return learning rate in case of the time 
    """
    max_factor = torch.distributed.get_world_size()
    factor = 1.0 + (max_factor - 1.0) *min(epoch_num/config['warmup_epochs'], 1.0)
    if config['name'] == "CNN" or config['name'] == 'cifar100' or config['name'] == 'svhn':
        if epoch_num <= 150:
            new_lr = config['init_lr']
            return new_lr
        elif epoch_num > 150 and epoch_num <=250:
            new_lr = config['init_lr']/10.0
            return new_lr
        elif epoch_num > 250:
            new_lr = config['init_lr']/100.0
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


def replace_grad_by_momentum(grad, momentum):
    """
    Inplace operation that applies momentum to a gradient.
    This distinguishes between types of momentum (heavy-ball vs nesterov)
    """
    grad[:] = momentum


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
        config = cifar_config
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
    #TODO: Fix this for distributed
    # use parameter groups to get things for different learning rates
    # and weight decay parameters 
    current_lr = config['init_lr']

    if args.compressor != "signum":
        if config['name'] == "CNN" or config['name'] == 'cifar100' or config['name'] == 'svhn':
            # optimizer only for langauge model
            # otherwise we are going manual\
            # my guess is that repackage thing for language models changes
            # the model structure and the optimizer is registered only for some of
            # the parameters
            optimizer = optim.SGD(train_task.model.parameters(), lr=current_lr,
                                            momentum=config['momentum'],
                                            weight_decay=0.0001)
        if config['name'] == 'imagenet':
            # parameters 
            parameters = add_weight_decay(train_task.model, config['weight_decay'])
            # weight decay is incorporated in the parameters
            optimizer = optim.SGD(parameters, lr=current_lr,
                                  momentum=config['momentum'], weight_decay=0)

            # let's comment out the learning rate warmup for now
            # scheduler_multi_step = lr_scheduler.MultiStepLR(
            #     optimizer, milestones=[e - config['warmup_epoch']- 1 for e in
            #                            config['lr_decay_period']],
            #     gamma=config['lr_decay_factor'])
            # scheduler_warmup = GradualWarmupScheduler(
            #     optimizer, multiplier=config['multiplier'],
            #     total_epoch=config['warmup_epoch'], after_scheduler=scheduler_multi_step)
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
    
    # since we are simulating things, we will need to allocate buffer for each simulated node
    vectorized_net = vectorize_grad([p.data for p in train_task.model.parameters()])
    momenta = []
    simulated_grad_buffer = torch.empty(args.num_simulated_nodes, vectorized_net.size()[0]).to(config['device']) # #simulated node X model dimention d
    for node_index in range(args.num_simulated_nodes):
        momenta.append([torch.empty_like(param) for param in train_task.model.parameters()])

    first_iter = 0 # hack for momentum code

    if args.compressor == "vanilla":
        grad_compressor = None
    elif args.compressor == "suquantization":
        grad_compressor = StochasticUniformQuantization(random_seed=0, device=config['device'])
    elif args.compressor == "signum":
        grad_compressor = SignSGDwithMajorityVoteReducerSimulation(random_seed=0, device=config['device'])
    else:
        raise NotImplementedError("Unsupported gradient compressor !")

    
    wds = [get_weight_decay(name, config) for name in train_task.parameter_names]
    for epoch in range(config['num_epochs']):
        # to put into the `breakdown_time_log`
        epoch_compute_time = 0.0
        epoch_comm_time = 0.0
        epoch_total_time = 0.0
        epoch_encoding_overhead = 0.0
        epoch_iter_time = 0.0

        # for logging out the current learning rate
        if args.compressor != "signum":
            for param_group in optimizer.param_groups:
                logger.info("### Epoch: {}, Current Effective lr: {}".format(epoch, param_group['lr']))
                break
        else:
            current_lr = get_lr(config=config, epoch_num=epoch)
            logger.info("### Epoch: {}, Current Effective lr: {}".format(epoch, current_lr))

        if config['is_distributed']:
            train_task.sampler.set_epoch(epoch) # set epoch to make sure the data is reshuffled per epoch
              
        torch.cuda.synchronize()  
        tic = time.time()
        elements_per_epoch = 0
        simulated_nodes_index = 0 # which is [0, args.num_simulated_nodes-1]
        global_iter = 0
 
        # note that currently we assume everything is running over CUDA
        train_task.model.train()

        
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

            if args.compressor == "signum":
                # based on the discussion in https://arxiv.org/pdf/1810.05291.pdf, 
                # momentum rather than gradient is compressed
                # we thus calculate the momentum first
                for grad, momentum in zip(grad_train, momenta[simulated_nodes_index]):
                    if epoch == 0 and iter_index < args.num_simulated_nodes:
                        momentum.data = grad.clone().detach()
                    else:
                        momentum.mul_(config["momentum"]).add_(
                                                                1 - config["momentum"], grad
                                                            )
                        replace_grad_by_momentum(grad, momentum)

            # aggregate the gradients here:
            concat_grad = vectorize_grad(grad_train)

            simulated_grad_buffer[simulated_nodes_index] = concat_grad
            # collect the gradient for simulated user
            simulated_nodes_index += 1

            if simulated_nodes_index < args.num_simulated_nodes:
                continue
            else:
                print("######## Epoch: {} | Global iter: {}/{}".format(epoch, global_iter, int(len(train_task.train_loader)/args.num_simulated_nodes)))
                # communication step
                if args.compressor == "vanilla":
                    comm_start.record()

                if config['grad_comb']:
                    if args.compressor == "vanilla":
                        torch.distributed.all_reduce(concat_grad, async_op=False)
                        concat_grad[:] = concat_grad/args.num_nodes
                    elif args.compressor == "suquantization":
                        #print("##### max: {}, min grad: {}".format(torch.max(concat_grad), torch.min(concat_grad)))
                        reduced_aggregated_grad, bits_communicated, compressor_iter_comm_time, iter_encode_decode_time = grad_compressor.reduce(simulated_grad_buffer)
                        concat_grad[:] = reduced_aggregated_grad/args.num_nodes
                    elif args.compressor == "signum":
                        #print("######## Compressing gradient local iter: {}".format(iter_index))
                        reduced_aggregated_grad, bits_communicated, compressor_iter_comm_time, iter_encode_decode_time = grad_compressor.reduce(simulated_grad_buffer)
                        concat_grad[:] = reduced_aggregated_grad
                    else:
                        raise NotImplementedError("Unsupported gradient compressor !")

                if args.compressor == "vanilla":
                    comm_end.record()
                    torch.cuda.synchronize()

                if config['grad_comb']:
                    out_grad_list = devectorize_grad(concat_grad, train_task.model)
                    if args.compressor == "vanilla":
                        iter_comm_cost = float(comm_start.elapsed_time(comm_end))/1000.0
                        epoch_comm_time += iter_comm_cost
                    elif args.compressor in ("suquantization", "signum"):
                        epoch_comm_time += compressor_iter_comm_time
                        epoch_encoding_overhead += iter_encode_decode_time
                    else:
                        raise NotImplementedError("Unsupported gradient compressor !")
                
                # updated the gradients in place
                # TODO: Move this to a new function
                if args.compressor != "signum":
                    for idx, param in enumerate(train_task.model.parameters()):
                        param.grad.data = out_grad_list[idx]
            
                    if config['name'] == 'CNN' or config['name'] == 'cifar100' or config['name'] == 'imagenet' or config['name'] == 'svhn':
                        optimizer.step()
                        optimizer.zero_grad()
                    else:
                        raise NotImplementedError("Unsupported model name type ...")
                else:
                    # for signsgd we will need to handle weight decay manually: (line ``update parameters"" in https://openreview.net/pdf?id=BJxhijAcY7)
                    for grad, param, wd in zip(out_grad_list, train_task.model.parameters(), wds):
                        if wd > 0:
                            grad.add_(wd, param.data.detach())

                    for grad, p in zip(out_grad_list, train_task.model.parameters()):
                        p.data.add_(-current_lr, grad)

                    train_task.model.zero_grad()

                iter_end.record()
                torch.cuda.synchronize()
                iter_total_dur = float(iter_start.elapsed_time(iter_end))/1000.0
                epoch_iter_time += iter_total_dur
                
                # meset the gradient buffer
                simulated_grad_buffer = torch.empty(args.num_simulated_nodes, vectorized_net.size()[0]).to(config['device'])
                simulated_nodes_index = 0
                global_iter += 1


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
        current_test_loss = train_task.validate_model(logger)

        current_test_loss = 10000
        if not best_test_loss or current_test_loss < best_test_loss:
            best_test_loss = current_test_loss

        if args.compressor != "signum":
            if config['name'] == 'CNN' or config['name'] == 'cifar100' or config['name'] == 'svhn':
                for group in optimizer.param_groups:
                    group['lr'] = current_lr
            if config['name'] == 'imagenet':
                scheduler_multi_step.step()
        else:
            pass


if __name__ == "__main__":
    # making sure seed is the first thing to be called
    seed(42)
    args = add_fit_args(argparse.ArgumentParser(description='Auto Scale'))
    log_file_name = os.path.basename(args.norm_file).split(".")[0] + ".log"
    logging.basicConfig(filename=log_file_name)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.info("Arguments: {}".format(args))
    print(args)
    #main(dataset="Cifar10", jl=True)
    main(args)
