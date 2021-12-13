import glob
import os
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms


def to_rgb(image):
    rgb_image = Image.new("RGB", image.size)
    rgb_image.paste(image)
    return rgb_image

class ImageDataset(Dataset):
    def __init__(self, root, input_shape, mode='train', target=2):
        # self.transform = transforms.Compose(transforms_)
        self.transform = transforms.Compose(
            [
                transforms.Resize(input_shape[-2:], Image.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )
        unaligned = False
        self.unaligned = unaligned
        self.mode = mode
        self.target = target
        self.input_shape = input_shape
        self.files_A = sorted(glob.glob(os.path.join(root, f"{self.mode}A") + "/*.*"))
        self.files_B = sorted(glob.glob(os.path.join(root, f"{self.mode}B") + "/*.*"))
        self.aug_func = transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=0.1)

    def __getitem__(self, index):
        path_A = self.files_A[index % len(self.files_A)]

        image_A = Image.open(path_A)
        path_A=(path_A.split("/")[-1]).split("_")[-3]
        # print(path_A)

        path_B = self.files_B[index % len(self.files_B)]
        image_B = Image.open(path_B)
        # Convert grayscale images to rgb
        if image_A.mode != "RGB":
            image_A = to_rgb(image_A)
        if image_B.mode != "RGB":
            image_B = to_rgb(image_B)

        if self.mode == 'test':
            aug_A = transforms.functional.affine(img=image_A, angle=15, translate=(0.1, 0.1), scale=(0.9), shear=0.1)
        else:
            aug_A = self.aug_func(image_A)
        img_tmp = np.asarray(image_B, dtype='uint8')
        blank_image = np.zeros((self.input_shape[1], self.input_shape[2], self.input_shape[0]), np.uint8)
        # target, _, area_p, _ =pp.critic_segmentation(img_tmp)
        label = self.target if np.sum(img_tmp) > np.sum(blank_image) else 0
        item_A = self.transform(image_A)
        item_B = self.transform(image_B)

        item_augA = self.transform(aug_A)

        return item_A, item_B,path_A


    def __len__(self):
        return max(len(self.files_A), len(self.files_B))