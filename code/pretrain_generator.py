from fastai.vision.learner import create_body
from torchvision.models.resnet import resnet18
from fastai.vision.models.unet import DynamicUnet

import os, sys
import numpy as np
import matplotlib.pyplot as plt
import tqdm

from PIL import Image

import torch
from torchvision import transforms
from torchvision.utils import make_grid
from torch.utils.data import Dataset, DataLoader

import glob
from fastai.data.external import untar_data, URLs

from skimage.color import rgb2lab, lab2rgb
from utils import showImageGrid, create_loss_meters, update_losses, log_results, visualize, lab_to_rgb
from dataset import GrayscaleColorizationDataset
from cgan import cGAN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
coco_path = untar_data(URLs.COCO_SAMPLE)
coco_path = str(coco_path) + "/train_sample"
print(coco_path)
#sys.exit()

np.random.seed(42)
paths = glob.glob(coco_path + "/*.jpg")
paths = np.random.choice(paths, 10000, replace = False)
print(len(paths))

rand_idxs = np.random.permutation(len(paths))
train_idxs = rand_idxs[ : int(np.floor(0.8*len(paths)))]
val_idxs = rand_idxs[int(np.floor(0.8*len(paths))) : ]

train_paths = np.asarray(paths)[train_idxs]
val_paths = np.asarray(paths)[val_idxs]
print(len(train_paths), len(val_paths))

train_dl = DataLoader(GrayscaleColorizationDataset(train_paths), 
                      batch_size = 16,
                      shuffle = True)


def build_res_unet(n_input=1, n_output=2, size=256):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    body = create_body(resnet18, pretrained=True, n_in=n_input, cut=-2)
    net_G = DynamicUnet(body, n_output, (size, size)).to(device)
    return net_G

test_transforms = transforms.Compose([
    transforms.Resize((256, 256),  Image.BICUBIC),
])

def img_load(img_path):

    img = Image.open(img_path).convert("RGB")
    img = np.array(test_transforms(img))
    img_lab = transforms.ToTensor()(rgb2lab(img).astype("float32"))
    L = img_lab[[0], ...] / 50. - 1. # Between -1 and 1
    ab = img_lab[[1, 2], ...] / 110. # Between -1 and 1

    L = torch.unsqueeze(L,0)
    ab = torch.unsqueeze(ab,0)
    print("shape: L,ab",L.shape,ab.shape)
    return {'L': L, 'ab': ab}

def test_model(model, epoch, test_data, save_dir):
    #torch.manual_seed(0)
    #np.random.seed(0)
    #model.load(epoch)
    model.generator.eval()

    output_dir = os.path.join(save_dir,"outputs")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for i, img in enumerate(test_data):
        data = img_load(img)

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
        ax.imshow(L[0][0].cpu(), cmap='gray')
        ax.axis("off")
        ax = plt.subplot(1, 3, 2)
        ax.imshow(fake_imgs[0])
        ax.axis("off")
        ax = plt.subplot(1, 3, 3)
        ax.imshow(real_imgs[0])
        ax.axis("off")
        #plt.show()

        save = True
        if save:
            fig.savefig(os.path.join(output_dir,f"result_{i}.png"))

def pretrain_generator(net_G, train_dl, opt, criterion, epochs):
    for e in range(epochs):
        loss_meter = AverageMeter()
        for data in tqdm(train_dl):
            L, ab = data['L'].to(device), data['ab'].to(device)
            preds = net_G(L)
            loss = criterion(preds, ab)
            opt.zero_grad()
            loss.backward()
            opt.step()
            
            loss_meter.update(loss.item(), L.size(0))
            
        print(f"Epoch {e + 1}/{epochs}")
        print(f"L1 Loss: {loss_meter.avg:.5f}")


save_dir = "results"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

    
net_G = build_res_unet(n_input=1, n_output=2, size=256)
#opt = optim.Adam(net_G.parameters(), lr=1e-4)
#criterion = nn.L1Loss()        
#pretrain_generator(net_G, train_dl, opt, criterion, 20)
#torch.save(net_G.state_dict(), "res18-unet.pt")



net_G.load_state_dict(torch.load("res18-unet.pt", map_location=device))
model = cGAN(net_G=net_G)
model.load_state_dict(torch.load("final_model_weights.pt", map_location=device))

print("Model weights loaded")
test_data = ["/home/pandotra/.fastai/data/coco_sample/train_sample/000000000030.jpg"]
#test_data = ["C:\\Users\\Himanshu\\.fastai\\data\\coco_sample\\train_sample\\000000000030.jpg"]
#"C:\\Users\\Himanshu\\.fastai\\data\\coco_sample\\train_sample\\000000000089.jpg"]
epoch = 0
test_model(model, epoch, test_data, save_dir)