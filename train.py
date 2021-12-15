import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import sys
# from model import ResNetUNet
from monai.networks.nets import UNet
from monai.losses import DiceCELoss, DiceLoss

from dataset import TrainDataset, TestDataset, test_transform
from model.losses import BCEDiceIoUWithLogitsLoss2d
from model.functional import convert_by_one_hot_nd


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default="train_set/", help='path to the train data folder')
    parser.add_argument('--mode', type=str, default='train', help='mode')
    parser.add_argument('--batch_size', type=int, default=48, help='batch size')
    parser.add_argument('--epoch', type=int, default=500, help='no. of epoch')
    # parser.add_argument('--LR', type=float, default=0.00001, help='learning rate')
    parser.add_argument('--LR', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--n_channel', type=int, default=3, help='input channel')
    parser.add_argument('--img_height', type=int, default=512, help='image height')
    parser.add_argument('--img_width', type=int, default=768, help='image width')
    parser.add_argument('--n_class', type=int, default=8, help='output classes')
    parser.add_argument('--split_ratio', type=float, default=0.8, help='train/val split ratio')
    parser.add_argument('--num_workers', type=int, default=12, help='number of data loader workers')
    return parser.parse_args(argv)


class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_shape = (args.n_channel, args.img_height, args.img_width)
    split_ratio = args.split_ratio
    dataset = TrainDataset(args.data_path, input_shape)
    tds, vds = random_split(
        dataset,
        (int(len(dataset) * split_ratio), len(dataset) - int(len(dataset) * split_ratio))
    )
    vds.transform = test_transform(input_shape)

    train_loader = DataLoader(
        tds, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True, shuffle=True)
    test_loader = DataLoader(
        vds, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False, pin_memory=True)
    # dataset = TestDataset(args.data_path, input_shape)

    model = UNet(spatial_dims=2, in_channels=args.n_channel, out_channels=args.n_class,
                 channels=(16, 32, 64, 128, 256), strides=(2, 2, 2, 2), num_res_units=2)
    # criterion = BCEDiceIoUWithLogitsLoss2d()
    criterion = DiceCELoss(include_background=True, softmax=True,
        lambda_dice=2.
    )
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.LR)
    # min_test_loss = 1e5
    min_test_loss = float('inf')
    epsilon = 1e-7

    print("starting training in", device)
    for j in range(args.epoch):
        print("***********************************************")
        print(" ")
        print(" ")
        model.train()
        criterion.train()
        cumulative_loss = 0.
        for i, (input, mask, _) in enumerate(train_loader):
            input = input.to(device)
            mask = mask.to(device).float()
            optimizer.zero_grad()
            outputs = model(input)

            loss = criterion(outputs, mask)
            print('iteration  == %d  epoch  == %d   loss  == %f ' % (i + 1, j + 1, loss))
            cumulative_loss += loss.item()
            loss.backward()
            optimizer.step()

        avg_loss = cumulative_loss / len(train_loader)
        print('Average loss after %d epoch is %f ' % ((j + 1, avg_loss)))

        model.eval()
        criterion.eval()
        cumulative_loss = 0.0
        
        sensitivity = torch.zeros(args.n_class).to(device)
        sensitivity_cnt = torch.zeros(args.n_class).to(device)

        with torch.no_grad():
            for i, (input, mask, labels) in enumerate(test_loader):
                input = input.to(device)
                mask = mask.to(device).float()
                optimizer.zero_grad()

                outputs = model(input)

                loss_test = criterion(outputs, mask)
                cumulative_loss += loss_test.item()
                outputs = convert_by_one_hot_nd(outputs, nd=2)
                # outputs = outputs.softmax(dim=-3)
                tp = (outputs * mask).sum(dim=(-2, -1))
                tp_fn = mask.sum(dim=(-2, -1))

                sensitivity += ((tp) / (tp_fn + epsilon)).sum(dim=0)
                sensitivity_cnt += (tp_fn != 0).sum(dim=0)
            
            sensitivity /= sensitivity_cnt
            print(sensitivity[:-1])

            avg_loss = cumulative_loss / len(test_loader)

            print('Average test loss after %d epoch is %f ' % ((j + 1, avg_loss)))

        if avg_loss < min_test_loss:
            if os.path.exists('model-laryngeal1_' + str(min_test_loss) + '.pth.tar'):
                os.remove('model-laryngeal1_' + str(min_test_loss) + '.pth.tar')
            torch.save(model.state_dict(), 'model-laryngeal1_' + str(avg_loss) + '.pth.tar')
            print("average loss decresed ......saving model")
            min_test_loss = avg_loss

        print("minimum average loss till now is %f" % (min_test_loss))


if __name__ == "__main__":
    main(parse_arguments(sys.argv[1:]))
