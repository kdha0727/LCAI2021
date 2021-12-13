import glob
import os
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2


def to_rgb(image):
    rgb_image = Image.new("RGB", image.size)
    rgb_image.paste(image)
    return rgb_image

class ImageDataset(Dataset):
    def __init__(self, root, input_shape, mode='train', target=2):

        if mode == 'train':
            self.transform = A.Compose(
                [
                    A.Resize(input_shape[1], input_shape[2]),
                    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
                    A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.5),
                    A.RandomBrightnessContrast(p=0.5),
                    A.Normalize(),
                    ToTensorV2(),
                ]
            )
        else:
            self.transform = A.Compose(
                [
                    A.Resize(input_shape[1], input_shape[2]),
                    A.Normalize(),
                    ToTensorV2(),
                ]
            )

        unaligned = False
        self.unaligned = unaligned
        self.mode = mode
        self.target = target
        self.input_shape = input_shape
        self.files_A = sorted(glob.glob(os.path.join(root, f"{self.mode}A") + "/*.*"))
        self.files_B = sorted(glob.glob(os.path.join(root, f"{self.mode}B") + "/*.*"))
        # self.aug_func = transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=0.1)

    def __getitem__(self, index):
        
        path_A = self.files_A[index % len(self.files_A)]
        image_A = Image.open(path_A)
        path_A=(path_A.split("/")[-1]).split("_")[-3]

        path_B = self.files_B[index % len(self.files_B)]
        image_B = Image.open(path_B)

        # Convert grayscale images to rgb
        if image_A.mode != "RGB":
            image_A = to_rgb(image_A)
        if image_B.mode != "RGB":
            image_B = to_rgb(image_B)

        # if self.mode == 'test':
        #     aug_A = transforms.functional.affine(img=image_A, angle=15, translate=(0.1, 0.1), scale=(0.9), shear=0.1)
        # else:
        #     aug_A = self.aug_func(image_A)

        # img_tmp = np.asarray(image_B, dtype='uint8')
        # blank_image = np.zeros((self.input_shape[1], self.input_shape[2], self.input_shape[0]), np.uint8)
        # target, _, area_p, _ =pp.critic_segmentation(img_tmp)
        # label = self.target if np.sum(img_tmp) > np.sum(blank_image) else 0

        image_A, image_B = np.array(image_A), np.array(image_B)
        augmented = self.transform(image=image_A, mask=image_B)
        item_A = Image.fromarray(augmented['image'])
        item_B = Image.fromarray(augmented['mask'])

        # item_augA = self.transform(aug_A)

        return item_A, item_B, path_A


    def __len__(self):
        return max(len(self.files_A), len(self.files_B))