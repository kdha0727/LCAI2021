import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import sys
# from model import ResNetUNet
from torch.nn import ModuleList
from monai.networks.nets import UNet
from monai.losses import DiceCELoss

from dataset import TestDataset, test_transform
from model.functional import convert_by_one_hot_nd
from model.efficientnet import get_efficientnet

from constants import class_to_index, img_width, img_height
from post_process import infer


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default="test_set_for_LCAI/", help='path to the train data folder')
    parser.add_argument('--epoch', type=int, default=500, help='no. of epoch')
    parser.add_argument('--n_class', type=int, default=8, help='output classes')
    parser.add_argument('--num_workers', type=int, default=8, help='number of data loader workers')
    parser.add_argument('--device', type=str, default='cuda', help='device type')
    return parser.parse_args(argv)


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    input_shape = (3, img_height, img_width)
    dataset = TestDataset(args.data_path, input_shape)
    dataset.transform = test_transform(input_shape)

    loader = DataLoader(
        dataset, batch_size=1, num_workers=4, pin_memory=True, shuffle=False)

    clf = get_efficientnet().to(device)
    clf.load_state_dict(torch.load('./weights/clf_efficientnet.pth.tar'))

    # min_test_loss = 1e5
    seg_models = ModuleList()
    for i, target in enumerate(['normal', 'benign_tumor', 'cancer']):

        model = UNet(spatial_dims=2, in_channels=3, out_channels=args.n_class,
                    channels=(16, 32, 64, 128, 256), strides=(2, 2, 2, 2), num_res_units=2)
        model.to(device)
        # min_test_loss = 1e5
        model.load_state_dict(torch.load('./weights/seg_{}.pth.tar'.format(target)))
        seg_models.append(model)

    model.eval()

    infer(clf, seg_models, loader, device)


if __name__ == "__main__":
    main(parse_arguments(sys.argv[1:]))
