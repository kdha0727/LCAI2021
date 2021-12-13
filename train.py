import os
import argparse
import torch
from torch.utils.data import DataLoader
import sys
from model import ResNetUNet
import pytorch_ssim

from dataset import ImageDataset
import torchvision.transforms as transforms

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path',type = str,default="training_data/"
                        , help='path to the main folder')
    parser.add_argument('--batch_size',type = int,default=15, help='batch size')
    parser.add_argument('--epoch',type = int,default=500, help='no. of epoch')
    parser.add_argument('--LR',type = int,default=0.00001, help='learning rate')
    parser.add_argument('--n_channel',type = int,default=3, help='input channel')
    parser.add_argument('--img_height',type = int,default=256, help='image height')
    parser.add_argument('--img_width',type = int,default=384, help='image width')
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
    inv_transform = transforms.Compose(
        [
            UnNormalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            transforms.ToPILImage(),
        ]
    )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_shape = (args.n_channel, args.img_height, args.img_width)
    dataset = ImageDataset(args.data_path , input_shape, mode='train')
    train_loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=24, pin_memory=True,shuffle=True)

    dataset = ImageDataset(args.data_path, input_shape, mode='test')
    test_loader=DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=24)

    model = ResNetUNet()
    criterian=pytorch_ssim.ssim
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.LR)
    min_test_loss = 100000
    print("starting training in",device)
    for j in range(args.epoch):
        show=True
        print("***********************************************")
        print(" ")
        print(" ")
        model.train()
        cumulative_loss=0.0
        for i, (input, mask,_) in enumerate(train_loader):
            input = input.to(device)
            mask = mask.to(device).float()
            optimizer.zero_grad()
            outputs = model(input)

            loss = 1.-criterian(mask, outputs)
            print('iteration  == %d  epoch  == %d   loss  == %f '%(i+1,j+1,loss))
            cumulative_loss += loss.item()
            loss.backward()
            optimizer.step()

        avg_loss = cumulative_loss / len(train_loader)
        print('Average loss after %d epoch is %f ' % ((j + 1, avg_loss)))

        model.eval()
        cumulative_loss = 0.0
        for i, (input, mask,_) in enumerate(test_loader):

            input = input.to(device)
            mask = mask.to(device).float()
            optimizer.zero_grad()

            outputs = model(input)

            loss_test = 1.-criterian(mask , outputs)
            cumulative_loss += loss_test.item()

        avg_loss = cumulative_loss / len(test_loader)
        print('Average test loss after %d epoch is %f ' % ((j + 1, avg_loss)))
        if avg_loss < min_test_loss:
            if os.path.exists('model-laryngeal1_' + str(min_test_loss) + '.pth.tar'):
                os.remove('model-laryngeal1_' + str(min_test_loss) + '.pth.tar')
            torch.save(model.state_dict(), 'model-laryngeal1_' + str(avg_loss) + '.pth.tar')
            print("average loss decresed ......saving model")
            min_test_loss = avg_loss
        print("minimum average loss till now is %f"%(min_test_loss))


if __name__ == "__main__":
    main(parse_arguments(sys.argv[1:]))


