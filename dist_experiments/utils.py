import re


def is_conv_param(parameter_name):
    """
    Says whether this parameter is a conv linear layer that 
    needs a different treatment from the other weights
    """
    return "conv" in parameter_name and "weight" in parameter_name


def is_batchnorm_param(parameter_name):
    """
    Is this parameter part of a batchnorm parameter?
    """
    return re.match(r""".*\.bn\d+\.(weight|bias)""", parameter_name)

def get_weight_decay(parameter_name, config):
    """Take care of differences between weight decay for parameters"""
    if is_conv_param(parameter_name):
        #print("@@@@@@@ param name: {}, is conv param !!!!!".format(parameter_name))
        return config["optimizer_weight_decay_conv"]
    elif is_batchnorm_param(parameter_name):
        #print("@@@@@@@ param name: {}, is bn param !!!!!".format(parameter_name))
        return config["optimizer_weight_decay_bn"]
    else:
        #print("@@@@@@@ param name: {}, is other param !!!!!".format(parameter_name))
        return config["optimizer_weight_decay_other"]