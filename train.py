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
    parser.add_argument('--data_path', type=str, default="train_set/", help='path to the train data folder')
    parser.add_argument('--target', type=str, default="all", help='class')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--epoch', type=int, default=500, help='no. of epoch')
    parser.add_argument('--LR', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--n_class', type=int, default=8, help='output classes')
    parser.add_argument('--split_ratio', type=float, default=0.9, help='train/val split ratio')
    parser.add_argument('--num_workers', type=int, default=8, help='number of data loader workers')
    parser.add_argument('--device', type=str, default='cuda', help='device type')
    return parser.parse_args(argv)


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    input_shape = (3, img_height, img_width)
    split_ratio = args.split_ratio
    if args.target != 'all':
        class_to_idx = {args.target: class_to_index[args.target]}
    else:
        class_to_idx = class_to_index.copy()
    dataset = TrainDataset(args.data_path, input_shape, class_to_idx)
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

    model = UNet(spatial_dims=2, in_channels=3, out_channels=args.n_class,
                 channels=(16, 32, 64, 128, 256), strides=(2, 2, 2, 2), num_res_units=2)
    # criterion = BCEDiceIoUWithLogitsLoss2d()
    criterion = DiceCELoss(include_background=True, softmax=True, lambda_dice=2.)
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
