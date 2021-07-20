import torch
from . import cifar_architectures
from torchvision import datasets, transforms
import torch.nn.functional as F


def decompose_weights(model, low_rank_model, rank_factor, arch):
    # SVD version
    reconstructed_aggregator = []
    print("Conducting model decomposition !!!!!!")
    if arch == "FullRankVGG19":
        for item_index, (param_name, param) in enumerate(model.state_dict().items()):
            if len(param.size()) == 4 and item_index not in range(0, 54):
                # resize --> svd --> two layer
                param_reshaped = param.view(param.size()[0], -1)
                rank = min(param_reshaped.size()[0], param_reshaped.size()[1])
                u, s, v = torch.svd(param_reshaped)

                sliced_rank = int(rank/rank_factor)
                u_weight = u * torch.sqrt(s)
                v_weight = torch.sqrt(s) * v
                u_weight_sliced, v_weight_sliced = u_weight[:, 0:sliced_rank], v_weight[:, 0:sliced_rank]

                u_weight_sliced_shape, v_weight_sliced_shape = u_weight_sliced.size(), v_weight_sliced.size()

                model_weight_v = u_weight_sliced.view(u_weight_sliced_shape[0],
                                                      u_weight_sliced_shape[1], 1, 1)
                
                model_weight_u = v_weight_sliced.t().view(v_weight_sliced_shape[1], 
                                                          param.size()[1], 
                                                          param.size()[2], 
                                                          param.size()[3])

                reconstructed_aggregator.append(model_weight_u)
                reconstructed_aggregator.append(model_weight_v)
            elif len(param.size()) == 2 and "classifier." in param_name and "classifier.6." not in param_name:
                #print(param_name, param.size())
                rank = min(param.size()[0], param.size()[1])
                u, s, v = torch.svd(param)
                sliced_rank = int(rank/rank_factor)
                u_weight = u * torch.sqrt(s)
                v_weight = torch.sqrt(s) * v
                u_weight_sliced, v_weight_sliced = u_weight[:, 0:sliced_rank], v_weight[:, 0:sliced_rank]

                model_weight_v = u_weight_sliced
                
                model_weight_u = v_weight_sliced.t()

                reconstructed_aggregator.append(model_weight_u)
                reconstructed_aggregator.append(model_weight_v)            
            else:
                reconstructed_aggregator.append(param)
                
                
        model_counter = 0
        reload_state_dict = {}
        for item_index, (param_name, param) in enumerate(low_rank_model.state_dict().items()):
            #print("#### {}, {}, recons agg: {}， param: {}".format(item_index, param_name, 
            #                                                                        reconstructed_aggregator[model_counter].size(),
            #                                                                       param.size()))
            assert (reconstructed_aggregator[model_counter].size() == param.size())
            reload_state_dict[param_name] = reconstructed_aggregator[model_counter]
            model_counter += 1

    elif arch == "ResNet18":
        for item_index, (param_name, param) in enumerate(model.state_dict().items()):
            if len(param.size()) == 4 and item_index not in range(0, 13) and ".shortcut." not in param_name:
                # resize --> svd --> two layer
                param_reshaped = param.view(param.size()[0], -1)
                rank = min(param_reshaped.size()[0], param_reshaped.size()[1])
                u, s, v = torch.svd(param_reshaped)

                sliced_rank = int(rank/rank_factor)
                u_weight = u * torch.sqrt(s)
                v_weight = torch.sqrt(s) * v
                u_weight_sliced, v_weight_sliced = u_weight[:, 0:sliced_rank], v_weight[:, 0:sliced_rank]

                u_weight_sliced_shape, v_weight_sliced_shape = u_weight_sliced.size(), v_weight_sliced.size()

                model_weight_v = u_weight_sliced.view(u_weight_sliced_shape[0],
                                                      u_weight_sliced_shape[1], 1, 1)
                
                model_weight_u = v_weight_sliced.t().view(v_weight_sliced_shape[1], 
                                                          param.size()[1], 
                                                          param.size()[2], 
                                                          param.size()[3])

                reconstructed_aggregator.append(model_weight_u)
                reconstructed_aggregator.append(model_weight_v)           
            else:
                reconstructed_aggregator.append(param)
                
        model_counter = 0
        reload_state_dict = {}
        for item_index, (param_name, param) in enumerate(low_rank_model.state_dict().items()):
            #print("#### {}, {}, recons agg: {}， param: {}".format(item_index, param_name, 
            #                                                                        reconstructed_aggregator[model_counter].size(),
            #                                                                       param.size()))
            assert (reconstructed_aggregator[model_counter].size() == param.size())
            reload_state_dict[param_name] = reconstructed_aggregator[model_counter]
            model_counter += 1
    else:
        raise NotImplementedError("Unsupported model arch ...")
    
    low_rank_model.load_state_dict(reload_state_dict)
    return low_rank_model


class cifarTrain(object):
    """
    Setup Cifar training, model config provides all the hyper parameters
    required

    model_config(dict): Dictionary of training config
    """
    def __init__ (self, model_config):
        self.device = model_config['device']
        self.model = self._create_model(model_config['arch'])

        self.__arch = model_config['arch']

        if model_config['training_mode'] in ("pufferfish", "powerfish"):
            self.rank_factor = model_config['rank_factor']
            self.lowrank_model = self._create_model(model_config['lowrank_arch'])
        
        # full train loader doesn't do sampling
        self.train_loader, self.test_loader, self.full_train_loader, self.sampler = self._create_data_loader(
            model_config['data_path'], model_config['num_dataloader_threads'],
        model_config['train_batch_size'], model_config['test_batch_size'],
            model_config['is_distributed'])
        self.criterion = torch.nn.CrossEntropyLoss().to(self.device)
        self.lr = model_config['init_lr']

        self.training_mode = model_config["training_mode"]
        if self.training_mode in ("pufferfish", "powerfish"):
            self.full_rank_warmup_epoch = model_config["full_rank_warmup_epoch"]

        self.parameter_names = [name for (name, _) in self.model.named_parameters()]

        if model_config['training_mode'] in ("pufferfish", "powerfish"):
            self.lowrank_parameter_names = [name for (name, _) in self.lowrank_model.named_parameters()]

    def _create_model(self, arch):
        """
        Returns the model skeleton of the specified architecture
        arch(string): Model architecture
        """
        #TODO: Fix this architecture thing
        model = getattr(cifar_architectures, arch)()
        model.to(self.device)
        return (model)

    def _create_data_loader(self, data_path, num_workers, train_batch_size,
                            test_batch_size, is_distributed):
        """
        Returns test and train loaders for a given dataset
        data_path(str): Location of dataset
        num_workers(int): Number of workers for loading data
        train_batch_size(int): Num images in training batch
        test_batch_size(int): Num images in test batch
        """
        transform_train = transforms.Compose([
                                transforms.RandomCrop(32, padding=4),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                            ])
        # data prep for test set
        transform_test = transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                                ])

        training_set = datasets.CIFAR10(root=data_path, train=True,
                                                    download=True, transform=transform_train)
        sampler = None
        is_shuffle = True
        if is_distributed:
            sampler = torch.utils.data.DistributedSampler(training_set)
            # when using sampler you don't use shuffle
            is_shuffle = False

        train_loader = torch.utils.data.DataLoader(training_set,
                                                   num_workers=num_workers,
                                                   batch_size=train_batch_size,
                                                   sampler = sampler,
                                                   shuffle=is_shuffle,
                                                   pin_memory=True)

        full_train_loader = torch.utils.data.DataLoader(training_set,
                                                        num_workers=num_workers,
                                                        batch_size=train_batch_size,
                                                        sampler=None,
                                                        shuffle=False,
                                                        pin_memory=True)

        test_set = datasets.CIFAR10(root=data_path, train=False,
                                    download=True, transform=transform_test)
        test_loader = torch.utils.data.DataLoader(test_set,
                                                  num_workers=num_workers,
                                                  batch_size=test_batch_size,
                                                  shuffle=False,
                                                  pin_memory=True)
        return (train_loader, test_loader, full_train_loader, sampler)

    def train_single_iter(self, epoch=None, logger=None, for_autoscale=False):
        """
        Train single iter and pack grads in a list and return that list
        """
        if not for_autoscale:
            self.model.train()
            train_data_loader = self.train_loader
        else:
            self.model.eval()
            train_data_loader = self.full_train_loader
        for batch_idx, (data, target) in enumerate(train_data_loader):
            data, target = data.to(self.device), target.to(self.device)
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            grad_array = [param.grad.data for param in self.model.parameters()]
            if batch_idx%4 == 0:
                if logger is not None:
                    # not called by autoscale routine
                    logger.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                                    epoch, batch_idx * len(data), len(self.train_loader.dataset),
                                    100. * batch_idx / len(self.train_loader), loss.item()))
            yield grad_array


    def batch_loss_and_gradient(self, batch_idx, data, target, logger=None, epoch=None):
        data, target = data.to(self.device), target.to(self.device)

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
                logger.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
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
        return test_loss

    def init_hybrid_net(self):
        self.lowrank_model = decompose_weights(model=self.model, 
                                low_rank_model=self.lowrank_model, 
                                rank_factor=self.rank_factor, 
                                arch=self.__arch)
