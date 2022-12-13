import os, sys
import numpy as np
import matplotlib.pyplot as plt
import tqdm

from PIL import Image

import torch
from torchvision import transforms
from torchvision.utils import make_grid
from torch.utils.data import Dataset, DataLoader

from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from training import lab_img_load, luv_img_load, ycc_img_load, gray_img_load, test_transforms
from utils import lab_to_rgb, luv_to_rgb, ycc_to_rgb

def test_model_lab(model, epoch, test_data):
    #torch.manual_seed(0)
    #np.random.seed(0)
    #model.load(epoch)
    model.net_G.eval()

    for i, img in enumerate(test_data):
        data = lab_img_load(img)

        with torch.no_grad():    
            model.setup_input(data)
            model.forward()
        
        fake_color = model.fake_color.detach()
        real_color = model.ab
        L = model.L
        fake_imgs = lab_to_rgb(L, fake_color)
        real_imgs = lab_to_rgb(L, real_color)
        
        fig = plt.figure(figsize=(15, 8))
        
        ax = plt.subplot(1, 3, 1)
        ax.imshow(real_imgs[0])
        ax.text(0, -15, "Original Image")
        ax.axis("off")
        ax = plt.subplot(1, 3, 2)
        ax.imshow(L[0][0].cpu(), cmap='gray')
        ax.text(0, -15, "Model Input")
        ax.axis("off")
        ax = plt.subplot(1, 3, 3)
        ax.imshow(fake_imgs[0])
        ax.text(0, -15, "Model Output")
        ax.axis("off")

        fake_imgs = torch.Tensor(fake_imgs.reshape(1,3,256,256))
        real_imgs = torch.Tensor(real_imgs.reshape(1,3,256,256))

        lpips = LearnedPerceptualImagePatchSimilarity(net_type='vgg')
        print(lpips(2 * fake_imgs - 1, 2 * real_imgs - 1))

def test_model_luv(model, epoch, test_data, save_dir):
    #torch.manual_seed(0)
    #np.random.seed(0)
    #model.load(epoch)
    model.net_G.eval()

    output_dir = os.path.join(save_dir,"outputs")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for i, img in enumerate(test_data):
        data = luv_img_load(img)

        with torch.no_grad():    
            model.setup_input(data)
            model.forward()
        
        fake_color = model.fake_color.detach()
        real_color = model.ab
        L = model.L
        fake_imgs = lab_to_rgb(L, fake_color)
        real_imgs = luv_to_rgb(L, real_color)
        
        fig = plt.figure(figsize=(15, 8))
        
        ax = plt.subplot(1, 3, 1)
        ax.imshow(real_imgs[0])
        ax.text(0, -15, "Original Image")
        ax.axis("off")
        ax = plt.subplot(1, 3, 2)
        ax.imshow(L[0][0].cpu(), cmap='gray')
        ax.text(0, -15, "Model Input")
        ax.axis("off")
        ax = plt.subplot(1, 3, 3)
        ax.imshow(fake_imgs[0])
        ax.text(0, -15, "Model Output")
        ax.axis("off")
        #plt.show()

def test_model_gray(model, epoch, test_data, save_dir):
    #torch.manual_seed(0)
    #np.random.seed(0)
    #model.load(epoch)
    model.net_G.eval()

    output_dir = os.path.join(save_dir,"outputs")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for i, img in enumerate(test_data):
        data = gray_img_load(img)
        img_rgb = Image.open(img).convert("RGB")

        #print(torch.min(data['L']), torch.max(img))

        with torch.no_grad():    
            model.setup_input(data)
            model.forward()
        
        fake_color = model.fake_color.detach()
        real_color = model.ab
        L = model.L
        fake_imgs = lab_to_rgb(L, fake_color)
        real_imgs = np.array(test_transforms(img_rgb))
        
        fig = plt.figure(figsize=(15, 8))
        
        ax = plt.subplot(1, 3, 1)
        ax.imshow(real_imgs[0])
        ax.text(0, -15, "Original Image")
        ax.axis("off")
        ax = plt.subplot(1, 3, 2)
        ax.imshow(L[0][0].cpu(), cmap='gray')
        ax.text(0, -15, "Model Input")
        ax.axis("off")
        ax = plt.subplot(1, 3, 3)
        ax.imshow(fake_imgs[0])
        ax.text(0, -15, "Model Output")
        ax.axis("off")
        #plt.show()

def test_model_ycc(model, epoch, test_data, save_dir):
    #torch.manual_seed(0)
    #np.random.seed(0)
    #model.load(epoch)
    model.net_G.eval()

    output_dir = os.path.join(save_dir,"outputs")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for i, img in enumerate(test_data):
        data = ycc_img_load(img)

        with torch.no_grad():    
            model.setup_input(data)
            model.forward()
        
        fake_color = model.fake_color.detach()
        real_color = model.ab
        L = model.L
        fake_imgs = lab_to_rgb(L, fake_color)
        real_imgs = ycc_to_rgb(L, real_color)
        
        fig = plt.figure(figsize=(15, 8))
        
        ax = plt.subplot(1, 3, 1)
        ax.imshow(real_imgs[0])
        ax.text(0, -15, "Original Image")
        ax.axis("off")
        ax = plt.subplot(1, 3, 2)
        ax.imshow(L[0][0].cpu(), cmap='gray')
        ax.text(0, -15, "Model Input")
        ax.axis("off")
        ax = plt.subplot(1, 3, 3)
        ax.imshow(fake_imgs[0])
        ax.text(0, -15, "Model Output")
        ax.axis("off")
        #plt.show()

def test_model(model, epoch, test_data, save_dir):
    model.net_G.eval()

    output_dir = os.path.join(save_dir,"outputs")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for i, img in enumerate(test_data):
        data_lab = lab_img_load(img)
        data_luv = luv_img_load(img)
        data_ycc = ycc_img_load(img)
        data_gray = gray_img_load(img)
        img_rgb = Image.open(img).convert("RGB")

        # Lab
        with torch.no_grad():    
            model.setup_input(data_lab)
            model.forward()

        fake_color_lab = model.fake_color.detach()
        real_color_lab = model.ab
        L_lab = model.L
        fake_imgs_lab = lab_to_rgb(L_lab, fake_color_lab)
        real_imgs_lab = lab_to_rgb(L_lab, real_color_lab)

        # Luv
        with torch.no_grad():    
            model.setup_input(data_luv)
            model.forward()

        fake_color_luv = model.fake_color.detach()
        real_color_luv = model.ab
        L_luv = model.L
        fake_imgs_luv = lab_to_rgb(L_luv, fake_color_luv)
        real_imgs_luv = luv_to_rgb(L_luv, real_color_luv)

        # YCbCr
        with torch.no_grad():    
            model.setup_input(data_ycc)
            model.forward()

        fake_color_ycc = model.fake_color.detach()
        real_color_ycc = model.ab
        L_ycc = model.L
        fake_imgs_ycc = lab_to_rgb(L_ycc, fake_color_ycc)
        real_imgs_ycc = ycc_to_rgb(L_ycc, real_color_ycc)

        # Grayscale
        with torch.no_grad():    
            model.setup_input(data_gray)
            model.forward()
        
        fake_color_gray = model.fake_color.detach()
        real_color_gray = model.ab
        L_gray = model.L
        fake_imgs_gray = lab_to_rgb(L_gray, fake_color_gray)
        real_imgs_gray = np.array(test_transforms(img_rgb))
        
        
        fig = plt.figure(figsize=(15, 8))
        
        ax = plt.subplot(1, 5, 1)
        ax.imshow(real_imgs_lab[0])
        ax.text(0, -15, "Original Colored Image")
        ax.axis("off")
        ax = plt.subplot(1, 5, 2)
        ax.imshow(fake_imgs_lab[0])
        ax.text(0, -15, "Model Output on Lab input")
        ax.axis("off")
        ax = plt.subplot(1, 5, 3)
        ax.imshow(fake_imgs_luv[0])
        ax.text(0, -15, "Model Output on Luv input")
        ax.axis("off")
        ax = plt.subplot(1, 5, 4)
        ax.imshow(fake_imgs_ycc[0])
        ax.text(0, -15, "Model Output on YCbCr input")
        ax.axis("off")
        ax = plt.subplot(1, 5, 5)
        ax.imshow(fake_imgs_gray[0])
        ax.text(0, -15, "Model Output on Weighted RGB Grayscale")
        ax.axis("off")

        real_imgs_lab = torch.Tensor(real_imgs_lab.reshape(1,3,256,256))
        fake_imgs_lab = torch.Tensor(fake_imgs_lab.reshape(1,3,256,256))
        fake_imgs_luv = torch.Tensor(fake_imgs_luv.reshape(1,3,256,256))
        fake_imgs_ycc = torch.Tensor(fake_imgs_ycc.reshape(1,3,256,256))
        fake_imgs_gray = torch.Tensor(fake_imgs_gray.reshape(1,3,256,256))

        lpips = LearnedPerceptualImagePatchSimilarity(net_type='vgg')
        print(lpips(2 * fake_imgs_lab - 1, 2 * real_imgs_lab - 1))
        print(lpips(2 * fake_imgs_luv - 1, 2 * real_imgs_lab - 1))
        print(lpips(2 * fake_imgs_ycc - 1, 2 * real_imgs_lab - 1))
        print(lpips(2 * fake_imgs_gray - 1, 2 * real_imgs_lab - 1))
