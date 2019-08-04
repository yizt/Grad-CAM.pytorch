# -*- coding: utf-8 -*-
"""
Created on 2019/8/4 上午9:53

@author: mick.yi

入口类

"""
import numpy as np
import torch
from torch import nn
from torchvision import models
import argparse
# import matplotlib.pyplot as plt
from skimage import io
import cv2
from interpretability.grad_cam import GradCAM
from interpretability.guided_back_propagation import GuidedBackPropagation


def get_net(net_name, weight_path=None):
    """

    :param net_name: 网络名称
    :param weight_path: 与训练权重路径
    :return:
    """
    pretrain = weight_path is None  # 没有指定权重路径，则加载默认的预训练权重
    if net_name in ['vgg', 'vgg16']:
        net = models.vgg16(pretrained=pretrain)
    elif net_name in ['resnet', 'resnet50']:
        net = models.resnet50(pretrained=pretrain)
    elif net_name in ['densenet', 'densenet121']:
        net = models.densenet121(pretrained=pretrain)
    elif net_name in ['inception']:
        net = models.inception_v3(pretrained=pretrain)
    else:
        raise ValueError('invalid network name:{}'.format(net_name))
    # 加载指定路径的权重参数
    if weight_path is not None:
        net.load_state_dict(torch.load(weight_path))
    return net


def get_last_conv_name(net):
    """
    获取网络的最后一个卷积层的名字
    :param net:
    :return:
    """
    layer_name = None
    for name, m in net.named_modules():
        if isinstance(m, nn.Conv2d):
            layer_name = name
    return layer_name


def prepare_image(image):
    image = image.copy()

    # 归一化
    means = np.array([0.485, 0.456, 0.406])
    stds = np.array([0.229, 0.224, 0.225])
    image -= means
    image /= stds

    image = np.ascontiguousarray(np.transpose(image, (2, 0, 1)))  # channel first
    image = torch.from_numpy(image)
    image.unsqueeze_(0)  # 增加batch维
    image = torch.tensor(image, requires_grad=True)
    return image


def save_cam(image, mask):
    """
    保存CAM图
    :param image: [H,W,C],原始图像
    :param mask: [H,W],范围0~1
    :return:
    """
    # mask转为heatmap
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    heatmap = heatmap[..., ::-1]  # gbr to rgb
    io.imsave('heatmap.jpg',heatmap)
    # 合并heatmap到原始图像
    cam = heatmap + np.float32(image)
    cam = cam / np.max(cam)
    io.imsave("cam.jpg", np.uint8(255 * cam))


def main(args):
    img = io.imread(args.image_path)
    img = np.float32(cv2.resize(img, (224, 224))) / 255
    inputs = prepare_image(img)

    net = get_net(args.network, args.weight_path)

    layer_name = get_last_conv_name(net) if args.layer_name is None else args.layer_name
    grad_cam = GradCAM(net, layer_name)

    mask = grad_cam(inputs, args.class_id)
    save_cam(img, mask)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--network', type=str, default='resnet50',
                        help='ImageNet classification network')
    parser.add_argument('--image-path', type=str, default='./examples/pic1.png',
                        help='input image path')
    parser.add_argument('--weight-path', type=str, default=None,
                        help='weight path of the model')
    parser.add_argument('--layer-name', type=str, default=None,
                        help='last convolutional layer name')
    parser.add_argument('--class-id', type=int, default=None,
                        help='class id')
    arguments = parser.parse_args()

    main(arguments)
