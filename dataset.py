import os
import numpy as np
from torch.utils.data import Dataset
from torchvision.datasets.folder import ImageFolder
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2


IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')

class_to_index = {
    'normal': 0,
    'benign_tumor': 1,
    'cancer': 2,
}


def train_transform(input_shape):
    return A.Compose(
        [
            A.Resize(input_shape[1], input_shape[2]),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
            A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            A.Normalize(),
            ToTensorV2(),
        ]
    )


def test_transform(input_shape):
    return A.Compose(
        [
            A.Resize(input_shape[1], input_shape[2]),
            A.Normalize(),
            ToTensorV2(),
        ]
    )


def to_rgb(image):
    rgb_image = Image.new("RGB", image.size)
    rgb_image.paste(image)
    return rgb_image


class TrainDataset(ImageFolder):  # train / val

    @staticmethod
    def make_dataset(directory, class_to_idx, extensions=IMG_EXTENSIONS, is_valid_file=None):
        instances = []
        directory = os.path.expanduser(directory)
        for target_class in sorted(class_to_idx.keys()):
            class_index = class_to_idx[target_class]
            target_dir = os.path.join(directory, target_class)
            if not os.path.isdir(target_dir):
                continue
            for root, _, fnames in sorted(os.walk(os.path.join(target_dir, 'image'), followlinks=True)):
                for fname in sorted(fnames):
                    path = os.path.join(root, fname)
                    if path.lower().endswith(extensions):
                        item = path, class_index
                        instances.append(item)
        return instances

    @staticmethod
    def _find_classes(*args, **kwargs):
        return list(class_to_index), class_to_index

    def __init__(self, root, input_shape):
        super().__init__(root)
        self.transform = train_transform(input_shape)

    def __getitem__(self, index):
        path, target = self.samples[index]
        with Image.open(path) as img:
            if img.mode != "RGB":  # Convert grayscale/alpha images to rgb
                # img = to_rgb(img)
                img = img.convert('RGB')
            image = np.asarray(img)  # noqa
        folder, filename = os.path.split(path)
        path_mask = os.path.join(
            os.path.dirname(folder), 'mask',
            os.path.splitext(filename)[0] + '.npy'
        )
        mask_load = np.load(path_mask)
        mask_background = np.zeros((mask_load.shape[0], mask_load.shape[1]), np.uint8)
        mask = np.zeros((mask_load.shape[0], mask_load.shape[1], 8), np.uint8)
        mask[:, :, :7] = mask_load
        mask_background[mask_load.sum(axis=2) == 0] = 1
        mask[:, :, 7] = mask_background
        augmented = self.transform(image=image, mask=mask)
        image = augmented['image']
        mask = augmented['mask'].permute(2, 0, 1)
        return image, mask, target

    def __len__(self):
        return len(self.samples)


class TestDataset(Dataset):  # test

    @staticmethod
    def make_dataset(directory, extensions=IMG_EXTENSIONS):
        instances = []
        directory = os.path.expanduser(directory)
        for root, _, fnames in sorted(os.walk(directory, followlinks=True)):
            for fname in sorted(fnames):
                path = os.path.join(root, fname)
                if path.lower().endswith(extensions):
                    instances.append(path)
        return instances

    def __init__(self, root, input_shape):
        self.samples = self.make_dataset(root)
        self.transform = test_transform(input_shape)

    def __getitem__(self, index):
        with Image.open(self.samples[index]) as img:
            if img.mode != "RGB":  # Convert grayscale/alpha images to rgb
                # img = to_rgb(img)
                img = img.convert('RGB')
            image = np.asarray(img)  # noqa
        augmented = self.transform(image=image)
        image = Image.fromarray(augmented['image'])
        return image

    def __len__(self) -> int:
        return len(self.samples)
