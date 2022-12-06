import os
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image

import torch
from torchvision import transforms
from torchvision.utils import make_grid
from torch.utils.data import Dataset, DataLoader

from utils import initialize_model
from generator import Generator
from discrimator import Discriminator
from loss import GANLoss


class cGAN(torch.nn.Module):
    def __init__(self, save_dir = "results", generator_lr = 2e-4, discriminator_lr = 2e-4, 
                 beta1 = 0.5, beta2 = 0.999, lambda_L1 = 100.):
        super().__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.result_dir = save_dir
        self.lambda_L1 = lambda_L1
        
        self.generator = initialize_model(Generator(c_i = 1, c_o = 2, num_filters=64), self.device)
        self.discriminator = initialize_model(Discriminator(c_i = 3, num_filters = 64), self.device)

        self.GANcriterion = GANLoss().to(self.device)
        self.L1criterion = torch.nn.L1Loss()
        
        self.generator_optimizer = torch.optim.Adam(self.generator.parameters(), 
                                                    lr = generator_lr, 
                                                    betas = (beta1, beta2))
        self.discriminator_optimizer = torch.optim.Adam(self.discriminator.parameters(), 
                                                        lr = discriminator_lr, 
                                                        betas = (beta1, beta2))
    
    def set_requires_grad(self, model, requires_grad = True):
        for p in model.parameters():
            p.requires_grad = requires_grad
        
    def setup_input(self, data):
        self.L = data['L'].to(self.device)
        self.ab = data['ab'].to(self.device)
        
    def forward(self):
        self.fake_color = self.generator(self.L)
    
    def backward_discriminator(self):
        fake_image = torch.cat([self.L, self.fake_color], dim = 1)
        fake_preds = self.discriminator(fake_image.detach())
        self.loss_discriminator_fake = self.GANcriterion(fake_preds, False)

        real_image = torch.cat([self.L, self.ab], dim = 1)
        real_preds = self.discriminator(real_image)
        self.loss_discriminator_real = self.GANcriterion(real_preds, True)

        self.loss_discriminator = (self.loss_discriminator_fake + self.loss_discriminator_real) * 0.5
        self.loss_discriminator.backward()
    
    def backward_generator(self):
        fake_image = torch.cat([self.L, self.fake_color], dim = 1)
        fake_preds = self.discriminator(fake_image)
        
        self.loss_generator_GAN = self.GANcriterion(fake_preds, True)
        self.loss_generator_L1 = self.L1criterion(self.fake_color, self.ab) * self.lambda_L1

        self.loss_generator = self.loss_generator_GAN + self.loss_generator_L1
        self.loss_generator.backward()
    
    def optimize(self):
        self.forward()
        self.discriminator.train()
        self.set_requires_grad(self.discriminator, True)
        self.discriminator_optimizer.zero_grad()
        self.backward_discriminator()
        self.discriminator_optimizer.step()
        
        self.generator.train()
        self.set_requires_grad(self.discriminator, False)
        self.generator_optimizer.zero_grad()
        self.backward_generator()
        self.generator_optimizer.step()

    def save(self, epoch):
        save_dir = os.path.join(self.result_dir)

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        torch.save(self.generator.state_dict(), os.path.join(save_dir, 'models_' + str(epoch) + "_" + 'G.pkl'))
        torch.save(self.discriminator.state_dict(), os.path.join(save_dir, 'models_' + str(epoch) + "_" + 'D.pkl'))

    def load(self, epoch):
        save_dir = os.path.join(self.result_dir)
        self.generator.load_state_dict(torch.load(os.path.join(save_dir, 'models_' + str(epoch) + "_" + 'G.pkl')))
        self.discriminator.load_state_dict(torch.load(os.path.join(save_dir, 'models_' + str(epoch) + "_" + 'D.pkl')))
