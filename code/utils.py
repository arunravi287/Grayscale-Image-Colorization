import os
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image

import torch
from torchvision import transforms
from torchvision.utils import make_grid
from torch.utils.data import Dataset, DataLoader

from skimage.color import rgb2lab, lab2rgb

def showImageGrid(train_paths):
  _, axes = plt.subplots(5, 5, figsize=(12, 12))
  for ax, img_path in zip(axes.flatten(), train_paths):
      ax.imshow(Image.open(img_path))
      ax.axis("off")

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

class AverageMeter:
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.count, self.avg, self.sum = [0.] * 3
    
    def update(self, val, count=1):
        self.count += count
        self.sum += count * val
        self.avg = self.sum / self.count

def create_loss_meters():
    loss_discriminator_fake = AverageMeter()
    loss_discriminator_real = AverageMeter()
    loss_discriminator = AverageMeter()
    loss_generator_GAN = AverageMeter()
    loss_generator_L1 = AverageMeter()
    loss_generator = AverageMeter()
    
    return {'loss_discriminator_fake': loss_discriminator_fake,
            'loss_discriminator_real': loss_discriminator_real,
            'loss_discriminator': loss_discriminator,
            'loss_generator_GAN': loss_generator_GAN,
            'loss_generator_L1': loss_generator_L1,
            'loss_generator': loss_generator}

def update_losses(model, loss_meter_dict, count):
    for loss_name, loss_meter in loss_meter_dict.items():
        loss = getattr(model, loss_name)
        loss_meter.update(loss.item(), count=count)

def lab_to_rgb(L, ab):
    """
    Takes a batch of images
    """
    
    L = (L + 1.) * 50.
    ab = ab * 110.
    Lab = torch.cat([L, ab], dim=1).permute(0, 2, 3, 1).cpu().numpy()
    rgb_imgs = []
    for img in Lab:
        img_rgb = lab2rgb(img)
        rgb_imgs.append(img_rgb)
    return np.stack(rgb_imgs, axis=0)
    
def visualize(model, data, save=True):
    model.generator.eval()
    with torch.no_grad():
        model.setup_input(data)
        model.forward()
    model.generator.train()
    fake_color = model.fake_color.detach()
    real_color = model.ab
    L = model.L
    fake_imgs = lab_to_rgb(L, fake_color)
    real_imgs = lab_to_rgb(L, real_color)
    fig = plt.figure(figsize=(15, 8))
    for i in range(5):
        ax = plt.subplot(3, 5, i + 1)
        ax.imshow(L[i][0].cpu(), cmap='gray')
        ax.axis("off")
        ax = plt.subplot(3, 5, i + 1 + 5)
        ax.imshow(fake_imgs[i])
        ax.axis("off")
        ax = plt.subplot(3, 5, i + 1 + 10)
        ax.imshow(real_imgs[i])
        ax.axis("off")
    plt.show()
    if save:
        fig.savefig(f"colorization_{time.time()}.png")
        
def log_results(loss_meter_dict):
    for loss_name, loss_meter in loss_meter_dict.items():
        print(f"{loss_name}: {loss_meter.avg:.5f}")