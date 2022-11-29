import os
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

from utils import showImageGrid, create_loss_meters, update_losses, log_results, visualize
from dataset import GrayscaleColorizationDataset
from cgan import cGAN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
coco_path = untar_data(URLs.COCO_SAMPLE)
coco_path = str(coco_path) + "/train_sample"

np.random.seed(42)
paths = glob.glob(coco_path + "/*.jpg")
paths = np.random.choice(paths, 3000, replace = False)
print(len(paths))

rand_idxs = np.random.permutation(len(paths))
train_idxs = rand_idxs[ : int(np.floor(0.7*len(paths)))]
val_idxs = rand_idxs[int(np.floor(0.7*len(paths))) : ]

train_paths = np.asarray(paths)[train_idxs]
val_paths = np.asarray(paths)[val_idxs]
print(len(train_paths), len(val_paths))

train_dl = DataLoader(GrayscaleColorizationDataset(train_paths), 
                      batch_size = 16,
                      shuffle = True)

val_dl = DataLoader(GrayscaleColorizationDataset(val_paths), 
                    batch_size = 16,
                    shuffle = True)

data = next(iter(train_dl))
L, ab = data['L'], data['ab']
print(L.shape, ab.shape)
print(len(train_dl), len(val_dl))

def train_model(model, train_dl, epochs, display_every=131):
    data = next(iter(val_dl)) # getting a batch for visualizing the model output after fixed intrvals
    for e in range(epochs):
        loss_meter_dict = create_loss_meters() # function returing a dictionary of objects to 
        i = 0                                  # log the losses of the complete network
        for data in tqdm.tqdm(train_dl):
            model.setup_input(data) 
            model.optimize()
            update_losses(model, loss_meter_dict, count=data['L'].size(0)) # function updating the log objects
            i += 1
            if i % display_every == 0:
                print(f"\nEpoch {e+1}/{epochs}")
                print(f"Iteration {i}/{len(train_dl)}")
                log_results(loss_meter_dict) # function to print out the losses
                visualize(model, data, save=False) # function displaying the model's outputs

model = cGAN()
train_model(model, train_dl, 20)