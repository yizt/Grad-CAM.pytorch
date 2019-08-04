# -*- coding: utf-8 -*-
"""
Created on 2019/8/4 上午9:45

@author: mick.yi

"""
import torch
from torch import nn
from torch.autograd import Function
import numpy as np


class _GuidedBackPropagationReLUFunction(Function):
    """
    定义Function实现自定义的forward和backward
    """

    def forward(self, x):
        output = torch.clamp(x, min=0)
        self.save_for_backward(x, output)
        return output

    def backward(self, grad_output):
        x, _ = self.saved_tensors
        forward_mask = (x > 0).type_as(x)
        backward_mask = (grad_output > 0.).type_as(grad_output)
        grad_input = grad_output * forward_mask * backward_mask
        return grad_input


class GuidedBackPropagationReLU(nn.Module):
    def __init__(self, **kwargs):
        super(GuidedBackPropagationReLU, self).__init__(**kwargs)

    def forward(self, x):
        return _GuidedBackPropagationReLUFunction().forward(x)


def replace_relu(m):
    """
    替换m中所有的ReLU为GuidedBackPropagationReLU
    :param m: module
    :return:
    """
    if len(m._modules) == 0:
        return

    for name, module in m._modules.items():
        if len(module._modules) > 0:
            replace_relu(module)
        elif isinstance(module, nn.ReLU):  # module是最基础的layer
            if isinstance(m, nn.Sequential):
                m[int(name)] = GuidedBackPropagationReLU()
            elif isinstance(m, nn.ModuleList):
                m[int(name)] = GuidedBackPropagationReLU()
            elif isinstance(m, nn.ModuleDict):
                m[name] = GuidedBackPropagationReLU()
            elif hasattr(m, name):
                setattr(m, name, GuidedBackPropagationReLU())


class GuidedBackPropagation(object):
    def __init__(self, net):
        self.net = net
        self.net.eval()

    def __call__(self, inputs, index=None):
        """

        :param inputs: [1,3,H,W]
        :param index: class_id
        :return:
        """
        output = self.net(inputs)  # [1,num_classes]
        if index is None:
            index = np.argmax(output.cpu().data.numpy())
        target = output[0][index]
        target.backward()

        return inputs.grad
