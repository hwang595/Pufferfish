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

from torch.optim import lr_scheduler
from warmup_scheduler import GradualWarmupScheduler

# added files
# import grad_utils
import train_network
import sparsify_gradient
import auto_scale
# commit to say powersgd wrap
auto_scale_high = 2
auto_scale_low = 1


imagenet_config = {
    "name" : "imagenet",
    "arch" : "resnet50_prune",
    "is_lowrank": False,
    "rank_factor": 4,
    "dataset" : "imagenet",
    "weight_decay": 0.0001,
    "device" : "cuda:0",
    "data_path": "/home/ubuntu/data",
    "num_dataloader_threads": 8,
    "train_batch_size": 128,
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
    "early_bird":True,
    "scratch":"./EBTrain-ImageNet/ResNet50/pruned_7008_0.7/pruned.pth.tar"
}


cifar_config = {
    "name" : "CNN",
    "arch" : "seresnet18",
    "dataset" : "Cifar10",
    "device" : "cuda:0",
    "data_path" : "./data/cifar10",
    "num_dataloader_threads" : 1,
    "train_batch_size" : 128,
    "test_batch_size" : 128,
    "init_lr" : 0.1,
    "momentum": 0.9,
    "num_epochs": 300,
    "decay_steps": [150, 200],
    "decay_factor" : [10, 100], # divide init lr with this
    "switch_freq" : 10,
    "warmup_epochs" : 5,  #for learning rate scheduling
}


cifar100_config = {
    "name" : "cifar100",
    "arch" : "vgg19",
    "dataset": "cifar100",
    "device": "cuda:0",
    "data_path": "./data/cifar100",
    "num_dataloader_threads": 2,
    "train_batch_size": 128,
    "test_batch_size": 128,
    "init_lr": 0.1,
    "momentum": 0.9, 
    "num_epochs": 300,
    "switch_freq": 10,
    "decay_factor": 10,
    "warmup_epochs": 5,
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
    parser.add_argument("--auto-switch", default=False, action="store_true",
                        help="Enables automatic switching")
    # the presence of fixed-k in args will make the value true
    parser.add_argument("--fixed-k", default=False, action="store_true",
                        help="Indicates if we want to use a fixed k")
    parser.add_argument("--k", default=None, type=int, 
                        help= "If fixed-k is true then uses this for training")
    parser.add_argument("--norm-file", type=str, 
                        default="wikitext_lstm_full_rank.json")
    parser.add_argument("--start-k", default=False, action="store_true",
                        help="starts with a k")
    parser.add_argument("--k-start", default=None, type= int,
                        help = "Fix the start k")
    parser.add_argument("--fixed-sched", default=False, action="store_true",
                        help="follow a fixed schedule")
    parser.add_argument("--zero-memory", default=False, action="store_true")

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
    if args.fixed_k and args.k is None:
        raise TypeError("args.k can't be none if args.fixed_k is enabled")
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

def get_lr(config, epoch_num, current_lr, best_test_loss, current_test_loss):
    """
    Return learning rate in case of the time 
    """
    max_factor = torch.distributed.get_world_size()
    factor = 1.0 + (max_factor - 1.0) *min(epoch_num/config['warmup_epochs'], 1.0)
    if config['name'] == "CNN" or config['name'] == 'cifar100' or config['name'] == 'svhn':
        # if epoch_num == 100:
            # new_lr = config['init_lr']
            # return new_lr
        # if epoch_num == 101:
            # new_lr = config['init_lr'] * 2.4
            # return new_lr
        # if epoch_num == 102:
            # new_lr = config['init_lr'] * 3.8
            # return new_lr
        # if epoch_num == 103:
            # new_lr = config['init_lr'] * 5.2
            # return new_lr
        # if epoch_num == 104:
            # new_lr = config['init_lr'] * 6.6
            # return new_lr
        if epoch_num <= 150:
            new_lr = config['init_lr'] *factor
            return new_lr
        elif epoch_num > 150 and epoch_num <=250:
            new_lr = config['init_lr']/10.0 *factor
            return new_lr
        elif epoch_num > 250:
            new_lr = config['init_lr']/100.0 *factor
            return new_lr
        else:
            print ("Something went wrong in learning rate selection")
    if config['name'] == 'imagenet':
        if epoch_num <= 30:
            new_lr = config['init_lr'] * factor
            return new_lr
        if epoch_num > 30 and epoch_num < 60:
            new_lr = config['init_lr']/10.0 * factor
            return new_lr
        if epoch_num > 60:
            new_lr = config['init_lr']/100.0 * factor
            return new_lr
    if config['name'] == 'languageModel' or config['name'] == 'newlanguageModel':
        #TODO: Verify this
        current_lr = config['init_lr'] * factor
        if epoch_num <=60:
            # anneal the rate
            # copied from powersgd
            return current_lr
        elif epoch_num > 60 and epoch_num < 80:
            return current_lr/10.0
        else:
            # no need to anneal the learning rate
            return current_lr/100.0
def get_lr_squeezenet(config, epoch_num):
    # regimes = [[0, 18, 5e-3, 5e-4],
               # [19, 29, 1e-3, 5e-4],
               # [30, 43, 5e-4, 5e-4],
               # [44, 52, 1e-4, 0],
               # [53, 1e8, 1e-5, 0]]

    regimes = [[0, 19, 5e-3, 5e-4],
               [20, 29, 1e-3, 5e-4],
               [30, 44, 5e-4, 5e-4],
               [45, 54, 1e-4, 0],
               [55, 1e8, 1e-5, 0]]
    for i, row in enumerate(regimes):
        if epoch_num >=row[0] and epoch_num <= row[1]:
            return (row[2], row[3]) #lr, weight decay

    print ("Epoch {} something wrong in get_lr_squeezenet".format(epoch_num))

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
    return torch.cat([grad_value.view(-1) for grad_value in grad_train])

def devectorize_grad(reduced_grad, model):
    out_grad_list = []
    index_bias = 0
    for p_index, p in enumerate(model.parameters()):
        out_grad_list.append(reduced_grad[index_bias:index_bias+p.numel()].view(p.size()))
        index_bias += p.numel()
    return out_grad_list


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
        config = imagenet_config
    elif args.model_type == "cifar100":
        config = cifar100_config
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


    # list containing applySparsify class collection
    # the applySparsify method will handle everything
    # None if no need for reduction for the corresponding
    sparsify_method = [sparsify_gradient.applySparsify(p.shape,
                                                       config['device']) if
                                                       p.ndimension() > 1 else
                                                       None for p in train_task.model.parameters()]
    
    # import ipdb; ipdb.set_trace()
    # Temporay to test code with fixed k
    if not args.fixed_k and not args.auto_switch:
        print ("Warning: Full Rank SGD being done")
    if args.fixed_k:
        print ("Chose a fixed k, k= {}".format(args.k))
        for m in sparsify_method:
            if m is not None:
                m.update_method(args.k, args.zero_memory)
            else:
                pass
    if args.start_k:
        print ("Starting with fixed k ={}".format(args.k_start))
        for m in sparsify_method:
            if m is not None:
                m.update_method(args.k_start, args.zero_memory)
            else:
                pass
    current_test_loss = None
    best_test_loss = None
    momenta = [torch.empty_like(param) for param in train_task.model.parameters()]
    first_iter = 0 # hack for momentum code

    
    for epoch in range(config['num_epochs']):
        # to put into the `breakdown_time_log`
        epoch_compute_time = 0.0
        epoch_comm_time = 0.0
        epoch_total_time = 0.0
        epoch_encoding_overhead = 0.0

        for param_group in optimizer.param_groups:
            logger.info("### Epoch: {}, Current Effective lr: {}".format(epoch, param_group['lr']))
            break

        if config['is_distributed']:
            train_task.sampler.set_epoch(epoch) # set epoch to make sure the data is reshuffled per epoch
        
        step_iter = train_task.train_single_iter(epoch=epoch, logger=logger,
                                                 for_autoscale=False)
        if args.fixed_sched:
            pass
              
        torch.cuda.synchronize()  
        tic = time.time()
        elements_per_epoch = 0
        # if epoch != 0:
            # print("Norm of gradients before starting {} at epoch {}".format([
                # torch.norm(l.grad.data).item() for l in train_task.model.parameters()]
                                                                            # ,epoch))
            # net = {
                # 'state': train_task.model.state_dict()
            # }
            # torch.save(net, "epoch_{}_before_training.pth".format(epoch))
        #full_rank_accum = [torch.zeros_like(copy_l) for copy_l in train_task.model.parameters()]
        
        # note that currently we assume everything is running over CUDA
        torch.cuda.synchronize()
        time_start_iter = time.time()
        iter_num = 0
        for grad_train in step_iter:
            iter_num += 1
            # TODO: Think carefully how you want to modify the gradients
            torch.cuda.synchronize()
            time_stop_iter = time.time()
            compute_time_per_dict[epoch][iter_num] = [time_start_iter,
                                                      time_stop_iter]
            epoch_compute_time += (time_stop_iter - time_start_iter)

            out_grad_list = list() #list to store output gradients


            # aggregate the gradients here:
            concat_grad = vectorize_grad(grad_train)

            # communication step
            torch.cuda.synchronize()
            iter_comm_start = time.time()

            if config['grad_comb']:
                torch.distributed.all_reduce(concat_grad, async_op=False)
                concat_grad[:] = concat_grad/args.num_nodes
            else:
                for idx, grad_val in enumerate(grad_train):
                    #full_rank_accum[idx].add_(grad_val.data)
                    sparse_object = sparsify_method[idx]
                    if sparse_object is not None:
                        out_grad_reduced, bytes_comm, comm_cost, overhead_cost = sparse_object.apply_method(
                        grad_val)
                        epoch_comm_time += comm_cost
                        epoch_encoding_overhead += overhead_cost
                        # out_grad_list.append(sparse_object.apply_method(grad_val))
                        out_grad_list.append(out_grad_reduced)
                        #for og_index, og in enumerate(out_grad_list):
                        #    logger.info("@@@@@ og_index: {}, og: {}".format(og_index,
                        #                        torch.norm(og).item()))
                        elements_per_epoch += bytes_comm
                    else:
                        # in case of distributed need to all reduce the singular
                        # values
                        if args.distributed:
                            elements_per_epoch += torch.numel(grad_val) 
                            torch.distributed.all_reduce(grad_val, 
                                                         async_op=False)
                            grad_val[:] = grad_val/args.num_nodes
                        out_grad_list.append(grad_val)

            torch.cuda.synchronize()
            iter_comm_end = time.time()

            if config['grad_comb']:
                out_grad_list = devectorize_grad(concat_grad, train_task.model)
                epoch_comm_time += (iter_comm_end - iter_comm_start)
            
            # updated the gradients in place
            # TODO: Move this to a new function
            for idx, param in enumerate(train_task.model.parameters()):
                param.grad.data = out_grad_list[idx]
                #logger.info("@@@@@ p_idx: {}, p_grad_data: {}".format(idx,
                #                                torch.norm(param.grad.data).item()))
            if config['name'] == 'CNN' or config['name'] == 'cifar100' or config['name'] == 'imagenet' or config['name'] == 'svhn':
                optimizer.step()
                optimizer.zero_grad()
            if config['name'] == "squeezenet_cifar":
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
            time_start_iter = time.time()
        torch.cuda.synchronize()
        toc = time.time()
        timing_log[epoch].append(tic)
        timing_log[epoch].append(toc)
        #breakdown_time_log_dict[epoch]["comp"] = epoch_compute_time
        #breakdown_time_log_dict[epoch]["comm"] = epoch_comm_time
        breakdown_time_log_dict[epoch] = {"comp":epoch_compute_time, "comm":epoch_comm_time, 
                                        "encdec_overhead":epoch_encoding_overhead, "total":toc-tic}
        floats_communicated[epoch] = elements_per_epoch
        #grad_calc_dict[epoch] = [torch.norm(pval).item() for pval in full_rank_accum]
        # dumping training method used every epoch
        # mostly for sanity checking
        # commenting out for future use
        
        # if epoch%10 == 0:
            # if args.rank == 0:
                # # net = {
                    # # 'state': train_task.model.state_dict()
                # # }
                # # torch.save(net, "./saved_model.pth")

                # # norm_list = train_task.get_train_norm("./saved_model.pth",
                                                      # # config)
                # # print (norm_list)
                # # grad_calc_dict[epoch] = norm_list
                # if epoch == 0:
                    # old_grad_norms = None
                # else:
                    # old_grad_norms = grad_calc_dict[epoch-10]
                # current_grad_norms = grad_calc_dict[epoch]
                # auto_scale_list, ratio_list =run_auto_scale(
                    # current_grad_norms, old_grad_norms, epoch)

                 
            # else:
                # time.sleep(5)
        torch.distributed.barrier()        
        method_array = list()
        for mth_sp in sparsify_method:
            if mth_sp is None:
                method_array.append("FullRank")
            elif mth_sp.k is None:
                method_array.append("FullRank")
            else:
                method_array.append(mth_sp.k)
        current_method_log[epoch] = method_array
        with open(current_method_log_fname, "w") as fout:
            json.dump(current_method_log, fout)
        with open(timing_log_fname, "w") as fout:
            json.dump(timing_log, fout)
        with open(bytes_log_fname, "w") as fout:
            json.dump(floats_communicated, fout)
        #with open(grad_calc_fname, "w") as fout:
        #    json.dump(grad_calc_dict, fout)
        with open(per_iteration_compute_time_log, "w") as fout:
            json.dump(compute_time_per_dict, fout)
        with open(breakdown_time_log_fname, "w") as fout:
            json.dump(breakdown_time_log_dict, fout)
        # import ipdb; ipdb.set_trace()
        if args.auto_switch:
            print("Auto switching enabled")
            if epoch % config['switch_freq'] == 0: 
                #TOD$O: Make acceptable k from args of config dict
                auto_scale_tensor = torch.zeros(
                    len(sparsify_method), device="cuda:0", dtype=torch.int32)
                if args.rank == 0:
                    # only doing it for master
                    #TODO: Make that 4 configurable
                    # ratio_val, prev_norm, auto_scale_per_layer = auto_scale.run_auto_scale_gng(train_task,
                                                                     # 4, args.norm_thresh, prev_norm)
                    
                    if epoch == 0:
                        old_grad_norms = None
                    else:
                        old_grad_norms = grad_calc_dict[epoch-config['switch_freq']]
                        # will give the previous grads 
                    current_grad_norms = grad_calc_dict[epoch]
                    auto_scale_per_layer, ratio_val = auto_scale.run_auto_scale_gng(
                        current_grad_norms, old_grad_norms, epoch)
                    # auto_scale_divergence_list = auto_scale.run_auto_scale_divergence(
                        # grad_calc_dict, epoch, config['num_epochs'],
                        # config['switch_freq'])
                    # if auto_scale_divergence_list is not None:
                        # for idx, value_in in enumerate(auto_scale_per_layer):
                            # auto_scale_per_layer[idx] = max(
                                # auto_scale_per_layer[idx],
                                # auto_scale_divergence_list[idx])
                    #CAUTION: Bad hack to dump values and test
                    # auto_scale_per_layer = [4]*len(auto_scale_tensor)
                    
                    print("Auto scale per layer calculated = {} at rank {}".format(
                        auto_scale_per_layer, args.rank))
                    # there could be None in auto_scale_per_layer
                    # to clean that up I use this map
                    #TODO: Add flags and condition checks for single machine
                    auto_scale_per_layer = list(map(lambda x: 999 if x==None else x,
                                           auto_scale_per_layer))
                    
                    auto_scale_tensor = torch.tensor(auto_scale_per_layer,
                                                     dtype=torch.int32).to(
                        'cuda:0')
                # broadcast autoscale values
                print ("Auto scale tensor before = {} for rank {}".format(
                    auto_scale_tensor, args.rank))
                torch.distributed.broadcast(auto_scale_tensor, 0)
                print ("Auto Scale Tensor after = {} for rank {}".format(
                    auto_scale_tensor, args.rank))
                auto_scale_per_layer = auto_scale_tensor.tolist()
                # substiuting None back
                auto_scale_per_layer = list(map(lambda x: None if x==999 else x,
                                           auto_scale_per_layer))
                print ("Auto scale list = {} for rank {}".format(
                    auto_scale_per_layer, args.rank))
                if args.rank == 0: 
                    candidate_method_stat[epoch] = prev_norm
                    ratio_calc_dict[epoch] = ratio_val
                for idx, spm in enumerate(auto_scale_per_layer):
                    chosen_method = auto_scale_per_layer[idx]
                    sparse_mth = sparsify_method[idx]
                    if sparse_mth is not None:
                        sparse_mth.update_method(chosen_method,
                                                 args.zero_memory)
                    else:
                        auto_scale_per_layer[idx] = None # so that json is clean
                chosen_method_log[epoch] = auto_scale_per_layer
                with open(json_f_name, "w") as fout:
                    json.dump(chosen_method_log, fout)
                with open(candidate_methods_stat_fname, "w") as fout:
                    json.dump(candidate_method_stat, fout)
                with open(ratio_log_fname, "w") as fout:
                    json.dump(ratio_calc_dict, fout)

        train_task.model.eval()
        current_test_loss = train_task.validate_model(logger)
        #current_test_loss = 10000
        if not best_test_loss or current_test_loss < best_test_loss:
            best_test_loss = current_test_loss
        # updating the learning rate
        prev_lr = current_lr
        if config['name'] != "squeezenet_cifar":
            pass
            # current_lr = get_lr(config, epoch, current_lr, best_test_loss,
                                # current_test_loss)
        else:
            current_lr, current_wd = get_lr_squeezenet(config, epoch)

        if current_lr < prev_lr:
            # Second rule of new auto scale
            # at decay point what to do
            if args.auto_switch:
                print ("Epoch {} deacy time making it k= {}".format(epoch,
                                                                    auto_scale_high))
                for m in sparsify_method:
                    if m is not None:
                        m.update_method(auto_scale_high)
                    else:
                        pass
        
        train_task.lr = current_lr # mostly for logging
        #TODO: Add one more logging to make sure that k is correct
        # this will read the sparsify method array and write out the 
        
        if config['name'] == 'CNN' or config['name'] == 'cifar100' or config['name'] == 'svhn':
            for group in optimizer.param_groups:
                group['lr'] = current_lr
        if config['name'] == "squeezenet_cifar":
            for group in optimizer.param_groups:
                group['lr'] = current_lr
                group['weight_decay'] = current_wd
        if config['name'] == 'imagenet':
            #scheduler_warmup.step()
            #scheduler_multi_step.step()
            scheduler_multi_step.step()


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
