import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import sys
# from model import ResNetUNet
from monai.networks.nets import UNet
from monai.losses import DiceCELoss

from dataset import TrainDataset, test_transform
from model.functional import convert_by_one_hot_nd

from constants import class_to_index, img_width, img_height


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', type=str, help='class')
    parser.add_argument('--data_path', type=str, default="train_set/", help='path to the train data folder')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--epoch', type=int, default=500, help='no. of epoch')
    parser.add_argument('--n_class', type=int, default=8, help='output classes')
    parser.add_argument('--num_workers', type=int, default=8, help='number of data loader workers')
    parser.add_argument('--device', type=str, default='cuda', help='device type')
    return parser.parse_args(argv)


def calculate_sensitivity(args):
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    input_shape = (3, img_height, img_width)
    class_to_idx = {args.target: class_to_index[args.target]}
    dataset = TrainDataset(args.data_path, input_shape, class_to_idx)
    dataset.transform = test_transform(input_shape)

    loader = DataLoader(
        dataset, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True, shuffle=True)
    # dataset = TestDataset(args.data_path, input_shape)

    model = UNet(spatial_dims=2, in_channels=3, out_channels=args.n_class,
                 channels=(16, 32, 64, 128, 256), strides=(2, 2, 2, 2), num_res_units=2)
    model.to(device)
    # min_test_loss = 1e5

    model.load_state_dict(torch.load('./weights/seg_{}.pth.tar'.format(args.target)))

    epsilon = 1e-7

    model.eval()

    sensitivity = torch.zeros(args.n_class).to(device)
    sensitivity_cnt = torch.zeros(args.n_class).to(device)

    with torch.no_grad():
        for i, (input, mask, labels) in enumerate(loader):
            input = input.to(device)
            mask = mask.to(device).float()

            outputs = model(input)

            outputs = convert_by_one_hot_nd(outputs, nd=2)
            # outputs = outputs.softmax(dim=-3)
            tp = (outputs * mask).sum(dim=(-2, -1))
            tp_fn = mask.sum(dim=(-2, -1))

            sensitivity += ((tp) / (tp_fn + epsilon)).sum(dim=0)
            sensitivity_cnt += (tp_fn != 0).sum(dim=0)

        sensitivity /= sensitivity_cnt
        print(sensitivity[:-1].cpu().numpy())


if __name__ == "__main__":
    calculate_sensitivity(parse_arguments(sys.argv[1:]))
