import os
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image

import torch
from torchvision import transforms
from torchvision.utils import make_grid
from torch.utils.data import Dataset, DataLoader

class UnetBlock(torch.nn.Module):
    def __init__(self, num_filters, ni, submodule = None, c_i = None,
                 use_dropout = False, innermost = False, outermost = False):
        super().__init__()
        self.outermost = outermost
        if c_i is None: c_i = num_filters

        downconv = torch.nn.Conv2d(c_i,
                                   ni, 
                                   kernel_size = 4,
                                   stride = 2,
                                   padding = 1,
                                   bias = False)
        
        downrelu = torch.nn.LeakyReLU(0.2, True)
        downnorm = torch.nn.BatchNorm2d(ni)
        uprelu = torch.nn.ReLU(True)
        upnorm = torch.nn.BatchNorm2d(num_filters)
        
        if outermost:
            upconv = torch.nn.ConvTranspose2d(ni * 2, 
                                              num_filters,
                                              kernel_size = 4,
                                              stride = 2, 
                                              padding = 1)
            down = [downconv]
            up = [uprelu, upconv, torch.nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = torch.nn.ConvTranspose2d(ni, 
                                              num_filters, 
                                              kernel_size = 4,
                                              stride = 2,
                                              padding = 1,
                                              bias = False)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = torch.nn.ConvTranspose2d(ni * 2,
                                              num_filters, 
                                              kernel_size = 4,
                                              stride = 2, 
                                              padding = 1, 
                                              bias = False)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]
            
            if use_dropout: 
              up += [torch.nn.Dropout(0.5)]
            
            model = down + [submodule] + up
        
        self.model = torch.nn.Sequential(*model)
    
    def forward(self, x):
        return self.model(x) if self.outermost else torch.cat([x, self.model(x)], 1)

class Generator(torch.nn.Module):
 
    def __init__(self, c_i, c_o, num_filters):
        super(Generator, self).__init__()

        # innermost block
        unet_block = UnetBlock(num_filters * 8, 
                               num_filters * 8, 
                               c_i = None,
                               submodule = None,
                               innermost = True)
 
        # intermediate blocks
        unet_block = UnetBlock(num_filters * 8, 
                               num_filters * 8, 
                               c_i = None, 
                               submodule = unet_block,
                               use_dropout = True)
        
        unet_block = UnetBlock(num_filters * 8, 
                               num_filters * 8, 
                               c_i = None, 
                               submodule = unet_block,
                               use_dropout = True)
        
        unet_block = UnetBlock(num_filters * 8, 
                               num_filters * 8, 
                               c_i = None, 
                               submodule = unet_block,
                               use_dropout = True)
        
        unet_block = UnetBlock(num_filters * 4, 
                               num_filters * 8, 
                               c_i = None, 
                               submodule = unet_block,
                               use_dropout = False)
        
        unet_block = UnetBlock(num_filters * 2, 
                               num_filters * 4, 
                               c_i = None, 
                               submodule = unet_block,
                               use_dropout = False)
        
        unet_block = UnetBlock(num_filters, 
                               num_filters * 2, 
                               c_i = None, 
                               submodule = unet_block,
                               use_dropout = False)
         
        # outermost block
        self.model = UnetBlock(c_o,
                               num_filters,
                               c_i = c_i,
                               submodule = unet_block,
                               outermost = True)  
 
    def forward(self, input):
        return self.model(input)
