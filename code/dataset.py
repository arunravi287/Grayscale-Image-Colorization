import os
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image

import torch
from torchvision import transforms
from torchvision.utils import make_grid
from torch.utils.data import Dataset, DataLoader
from skimage.color import rgb2lab, lab2rgb

class DatasetGrayscaleColorization(Dataset):
    def __init__(self, paths):
        self.transforms = transforms.Compose([
            transforms.Resize((256, 256),  Image.BICUBIC),
        ])
        self.paths = paths
    
    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        img = np.array(self.transforms(img))
        img_lab = transforms.ToTensor()(rgb2lab(img).astype("float32"))
        L = img_lab[[0], ...] / 50. - 1.
        ab = img_lab[[1, 2], ...] / 110.
        
        return {'L': L, 'ab': ab}
    
    def __len__(self):
        return len(self.paths)