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
                    batch_size = 4,
                    shuffle = False)

data = next(iter(train_dl))
L, ab = data['L'], data['ab']
print(L.shape, ab.shape)
print(len(train_dl), len(val_dl))

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
    torch.manual_seed(0)
    np.random.seed(0)
    model.load(epoch)
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


def train_model(model, train_dl, epochs, save_dir, display_every=10):
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
                print(f"\nEpoch {e}/{epochs}")
                print(f"Iteration {i}/{len(train_dl)}")
                log_results(loss_meter_dict, save_dir, e, i) # function to print out the losses
                visualize(model, data, save=False) # function displaying the model's outputs
            
        print(f"\n Done Epoch {e}")
        if (e % 5) == 0:
            model.save(e)
            print(f"\nModel Saved : Epoch {e}")


save_dir = "results"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

model = cGAN(save_dir)
train_model(model, train_dl, 20, save_dir)

#List of image paths 
#test_data = [im1, im2, im3...]
#test_data = ["C:\\Users\\Himanshu\\.fastai\\data\\coco_sample\\train_sample\\000000000030.jpg",
#"C:\\Users\\Himanshu\\.fastai\\data\\coco_sample\\train_sample\\000000000089.jpg"]
#epoch = 2
#test_model(model, epoch, test_data, save_dir)


