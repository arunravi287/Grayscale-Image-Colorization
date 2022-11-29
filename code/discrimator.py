import os
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image

import torch
from torchvision import transforms
from torchvision.utils import make_grid
from torch.utils.data import Dataset, DataLoader

class Discriminator(torch.nn.Module):
    def __init__(self, c_i, num_filters = 64):
        super().__init__()
        
        model = [self.get_layers(c_i, 
                                 num_filters, 
                                 normalization = False)]

        model += [self.get_layers(num_filters * 2 ** 0, 
                                  num_filters * 2 ** 1, 
                                  stride = 2)]  

        model += [self.get_layers(num_filters * 2 ** 1, 
                                  num_filters * 2 ** 2, 
                                  stride = 2)] 

        model += [self.get_layers(num_filters * 2 ** 2, 
                                  num_filters * 2 ** 3, 
                                  stride = 1)]                  

        model += [self.get_layers(num_filters * 2 ** 3, 
                                  1, 
                                  stride = 1, 
                                  normalization = False, 
                                  activation = False)]

        self.model = torch.nn.Sequential(*model)                                                   
        
    def get_layers(self, c_i, num_filters, 
                   kernel_size = 4, stride = 2, padding = 1, 
                   normalization = True, activation = True):
        layers = [torch.nn.Conv2d(c_i, 
                                  num_filters, 
                                  kernel_size, 
                                  stride, 
                                  padding, 
                                  bias = not normalization)]

        if normalization: 
          layers += [torch.nn.BatchNorm2d(num_filters)]

        if activation: 
          layers += [torch.nn.LeakyReLU(0.2, True)]

        return torch.nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)