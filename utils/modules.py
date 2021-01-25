import torch
import torch.nn as nn

from pyro.distributions.util import broadcast_shape

class Exp(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return torch.exp(x)

class ConcatModule(nn.Module):
    """
    a custom module for concatenation of tensors
    """
    def __init__(self):
        super().__init__()

    def forward(self, *input_args):
        # we have a single object
        if len(input_args) == 1:
            # regardless of type,
            # we don't care about single objects
            # we just index into the object
            input_args = input_args[0]

        # don't concat things that are just single objects
        if torch.is_tensor(input_args):
            return input_args
        else:
            shape = broadcast_shape(*[s.shape[:-1] for s in input_args]) + (-1,)
            input_args = [s.expand(shape) for s in input_args]
            return torch.cat(input_args, dim=-1)

class LModule(nn.ModuleList):
    """When forwarding, list of output for each module's will be created.   

    Parameters
    ----------
    module_list : list
        list of nn.Module 
    """
    def __init__(self, module_list):
        super().__init__(module_list)

    def forward(self, x):
        return [module(x) for module in  self]

class MLP(nn.Module):
    """For encoder_y, encoder_z and decoder

    Parameters
    ----------
    input_dim : int
        For encoder_y, it will be x's dim.
        For encoder_z, it will be x+y's dim.
        For decoder, it will be z+y's dim.

    hidden_dim : int

    output_dim : int or list
        For one-parameter model like Bernoulli and MultiCat, it will be int type.
        For two-parameter model like Gaussian, it will be list type.

    output_act : nn.Module class
        ex: nn.Sigmoid or nn.Softmax or list of nn.Module
    """
    def __init__(self, input_dim, hidden_dim, output_dim, output_act):
        super().__init__()
        
        all_modules = []
        all_modules.append(ConcatModule())
        all_modules.append(nn.Linear(input_dim, hidden_dim))
        all_modules[-1].weight.data.normal_(0, 0.001)
        all_modules[-1].bias.data.normal_(0, 0.001)
        all_modules.append(nn.ReLU())

        if type(output_dim) is int:
            all_modules.append(nn.Linear(hidden_dim, output_dim))
            all_modules[-1].weight.data.normal_(0, 0.001)
            all_modules[-1].bias.data.normal_(0, 0.001)
            all_modules.append(output_act())
        else:

            module_list = []
            for o_dim, o_act in zip(output_dim, output_act):
                sequential = []
                sequential.append(nn.Linear(hidden_dim, o_dim))
                sequential[-1].weight.data.normal_(0, 0.001)
                sequential[-1].bias.data.normal_(0, 0.001)
                if o_act is not None:
                    sequential.append(o_act())
                module_list.append(nn.Sequential(*sequential))

            all_modules.append(LModule(module_list))

        self.sequential_mlp = nn.Sequential(*all_modules)
        self.cuda()

    def forward(self, *args):
        return self.sequential_mlp(args)