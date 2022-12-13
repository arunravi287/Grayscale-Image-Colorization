import os
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image

import torch
from torchvision import transforms
from torchvision.utils import make_grid
from torch.utils.data import Dataset, DataLoader
import time

from skimage.color import lab2rgb, luv2rgb, gray2rgb, ycbcr2rgb

def initialzie_weights(net):
    def initialize(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and 'Conv' in classname:
            torch.nn.init.normal_(m.weight.data, 
                                  mean = 0.0, 
                                  std = 0.02)
            
            if hasattr(m, 'bias') and m.bias is not None:
                torch.nn.init.constant_(m.bias.data, 0.0)
        elif 'BatchNorm2d' in classname:
            torch.nn.init.normal_(m.weight.data, 1., 0.02)
            torch.nn.init.constant_(m.bias.data, 0.)
            
    net.apply(initialize)
    return net

def initialize_model(model, device):
    model = model.to(device)
    model = initialzie_weights(model)
    return model

def lab_to_rgb(L, ab):
    L = (L + 1.) * 50.
    ab = ab * 110.
    Lab = torch.cat([L, ab], dim=1).permute(0, 2, 3, 1).cpu().numpy()
    rgb_imgs = []
    for img in Lab:
        img_rgb = lab2rgb(img)
        rgb_imgs.append(img_rgb)
    return np.stack(rgb_imgs, axis=0)

def luv_to_rgb(L, uv):
    L = (L + 1.) * 50.
    uv = uv * 110.
    Luv = torch.cat([L, uv], dim=1).permute(0, 2, 3, 1).cpu().numpy()
    rgb_imgs = []
    for img in Luv:
        img_rgb = luv2rgb(img)
        rgb_imgs.append(img_rgb)
    return np.stack(rgb_imgs, axis=0)

def gray_to_rgb(L, ab):
    L = (L + 1.) * 50.
    ab = ab * 110.
    Lab = torch.cat([L, ab], dim=1).permute(0, 2, 3, 1).cpu().numpy()
    rgb_imgs = []
    for img in Lab:
        img_rgb = gray2rgb(img)
        rgb_imgs.append(img_rgb)
    return np.stack(rgb_imgs, axis=0)

def ycc_to_rgb(Y, cc):
    Y = (Y + 1.) * 50.
    cc = cc * 110.
    Ycc = torch.cat([Y, cc], dim=1).permute(0, 2, 3, 1).cpu().numpy()
    rgb_imgs = []
    for img in Ycc:
        img_rgb = ycbcr2rgb(img)
        rgb_imgs.append(img_rgb)
    return np.stack(rgb_imgs, axis=0)