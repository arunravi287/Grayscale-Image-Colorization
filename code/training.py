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

from skimage.color import rgb2lab, rgb2luv, rgb2gray, rgb2ycbcr
from dataset import DatasetGrayscaleColorization
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

train_dl = DataLoader(DatasetGrayscaleColorization(train_paths), 
                      batch_size = 16,
                      shuffle = True)

val_dl = DataLoader(DatasetGrayscaleColorization(val_paths), 
                    batch_size = 4,
                    shuffle = False)

data = next(iter(train_dl))
L, ab = data['L'], data['ab']
print(L.shape, ab.shape)
print(len(train_dl), len(val_dl))

test_transforms = transforms.Compose([
    transforms.Resize((256, 256),  Image.BICUBIC),
])

def lab_img_load(img_path):
    img = Image.open(img_path).convert("RGB")
    img = np.array(test_transforms(img))
    img_lab = transforms.ToTensor()(rgb2lab(img).astype("float32"))
    L = img_lab[[0], ...] / 50. - 1. # Between -1 and 1
    ab = img_lab[[1, 2], ...] / 110. # Between -1 and 1

    L = torch.unsqueeze(L,0)
    ab = torch.unsqueeze(ab,0)
    print("shape: L,ab",L.shape,ab.shape)
    return {'L': L, 'ab': ab}

def luv_img_load(img_path):
    img = Image.open(img_path).convert("RGB")
    img = np.array(test_transforms(img))
    img_luv = transforms.ToTensor()(rgb2luv(img).astype("float32"))
    L = img_luv[[0], ...] / 50. - 1. # Between -1 and 1
    uv = img_luv[[1, 2], ...] / 110. # Between -1 and 1

    L = torch.unsqueeze(L,0)
    uv = torch.unsqueeze(uv,0)
    print("shape: L,uv",L.shape,uv.shape)
    return {'L': L, 'ab': uv}

def gray_img_load(img_path):
    img = Image.open(img_path).convert("RGB")
    img = np.array(test_transforms(img))
    img_gray = transforms.ToTensor()(rgb2gray(img).astype("float32"))

    print(torch.min(img_gray),torch.max(img_gray))
    print(img_gray)
    img_gray = img_gray * 2.0 - 1.0

    img_luv = transforms.ToTensor()(rgb2luv(img).astype("float32"))
    # L = img_luv[[0], ...] / 50. - 1. # Between -1 and 1
    uv = img_luv[[1, 2], ...] / 110. # Between -1 and 1

    # L = torch.unsqueeze(L,0)
    uv = torch.unsqueeze(uv,0)
    # print("shape: L,uv",L.shape,uv.shape)
    return {'L': img_gray.unsqueeze(0),'ab':uv}
  
def ycc_img_load(img_path):
    img = Image.open(img_path).convert("RGB")
    img = np.array(test_transforms(img))
    img_luv = transforms.ToTensor()(rgb2ycbcr(img).astype("float32"))
    y = ((img_luv[0] - 16)/219) * 2 - 1
    cc = img_luv[[1, 2], ...] / 110. # Between -1 and 1

    y = torch.unsqueeze(y,0)
    y = torch.unsqueeze(y,0)
    cc = torch.unsqueeze(cc,0)
    print("shape: L,uv",y.shape,cc.shape)
    return {'L': y, 'ab': cc}


def train_model(model, train_dl, epochs, save_dir, display_every=100):
    data = next(iter(val_dl)) # getting a batch for visualizing the model output after fixed intrvals
    for e in range(epochs):
        i = 0                                  # log the losses of the complete network
        for data in tqdm.tqdm(train_dl):
            model.setup_input(data) 
            model.optimize()
            i += 1
            if i % display_every == 0:
                print(f"\nEpoch {e}/{epochs}")
                print(f"Iteration {i}/{len(train_dl)}")
            
        print(f"\n Done Epoch {e}")
        if (e % 10) == 0:
            model.save(e)
            print(f"\nModel Saved : Epoch {e}")


save_dir = "results"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

model = cGAN(save_dir)
train_model(model, train_dl, 100, save_dir)

#List of image paths 
#test_data = [im1, im2, im3...]
#test_data = ["C:\\Users\\Himanshu\\.fastai\\data\\coco_sample\\train_sample\\000000000030.jpg",
#"C:\\Users\\Himanshu\\.fastai\\data\\coco_sample\\train_sample\\000000000089.jpg"]
#epoch = 2
#test_model(model, epoch, test_data, save_dir)


