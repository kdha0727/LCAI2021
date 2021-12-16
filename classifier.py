import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.transforms import ToTensor

import albumentations as A
from albumentations.pytorch import ToTensorV2

from PIL import Image
import numpy as np

import sys
import os
import copy

from dataset import TrainDataset, train_transform, test_traansform
from models.efficientnet import get_efficientnet

from easyrun import Easyrun

image_shape = (3, 256, 384)


ds = TrainDataset(
    './LCAI2021/data_processed/train_set',
    input_shape = 
)

tds, vds = random_split(ds, (int(len(ds) * 0.9), len(ds) - int(len(ds) * 0.9)))
train_indices = tds.indices
val_indices = vds.indices
tds = copy.deepcopy(ds)
vds = copy.deepcopy(ds)
tds.samples = [ds.samples[i] for i in train_indices]
vds.samples = [ds.samples[i] for i in val_indices]
tds.transform = train_transform(image_shape)
vds.transform = test_transform(image_shape)
tdl = DataLoader(tds, batch_size=64, num_workers=8, shuffle=True)
vdl = DataLoader(vds, batch_size=32, num_workers=8, shuffle=False)

model = get_efficientnet(pretrained=True)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.99)
step_func = scheduler.step
epochs = 160

with Easyrun(
        model, 'CrossEntropyLoss', optimizer, epochs,
        tdl, vdl, None, log_interval=1, step_task=step_func,
        verbose=True, timer=True, snapshot_dir='.',
) as trainer:
    trainer.to(device)
    trainer()
