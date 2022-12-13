import os, sys
import numpy as np
import matplotlib.pyplot as plt
import tqdm

from PIL import Image

import torch
from torchvision import transforms
from torchvision.utils import make_grid
from torch.utils.data import Dataset, DataLoader

from skimage.color import rgb2lab, rgb2luv, rgb2gray, rgb2ycbcr
from training import test_transforms

test_data = [""] # enter path to image

img = Image.open(test_data[0]).convert("RGB")
img = np.array(test_transforms(img))

img_lab = transforms.ToTensor()(rgb2lab(img).astype("float32"))
L1 = img_lab[[0], ...] / 50. - 1.
ab = img_lab[[1, 2], ...] / 110.

L1 = torch.unsqueeze(L1,0)
ab = torch.unsqueeze(ab,0)

img_luv = transforms.ToTensor()(rgb2luv(img).astype("float32"))
L2 = img_luv[[0], ...] / 50. - 1.
uv = img_luv[[1, 2], ...] / 110.

L2 = torch.unsqueeze(L2,0)
uv = torch.unsqueeze(uv,0)

img_ycc = transforms.ToTensor()(rgb2ycbcr(img).astype("float32"))
L3 = img_ycc[0]
L3 = ((L3 - 16)/219) * 2 - 1

print(torch.min(L3), torch.max(L3))

# L3 = torch.unsqueeze(L3,0)
# cc = torch.unsqueeze(cc,0)

print(torch.min(L3))

img_gray = transforms.ToTensor()(rgb2gray(img).astype("float32"))
img_gray = img_gray * 2 - 1

fig = plt.figure(figsize=(15, 8))
        
ax = plt.subplot(1, 5, 1)
ax.imshow(img)
ax.text(0, -15, "Original Image")
ax.axis("off")
ax = plt.subplot(1, 5, 2)
ax.imshow(L1[0][0].cpu(), cmap='gray')
ax.text(0, -15, "Channel L from Lab")
ax.axis("off")
ax = plt.subplot(1, 5, 3)
ax.imshow(L2[0][0].cpu(), cmap='gray')
ax.text(0, -15, "Channel L from Luv")
ax.axis("off")
ax = plt.subplot(1, 5, 4)
ax.imshow(L3.cpu(), cmap='gray')
ax.text(0, -15, "Channel Y from YCbCr")
ax.axis("off")
ax = plt.subplot(1, 5, 5)
ax.imshow(img_gray.squeeze(0), cmap='gray')
ax.text(0, -15, "Weighted RGB Grayscale")
ax.axis("off")

