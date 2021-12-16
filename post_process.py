import torch

from constants import channel_map
from model.functional import convert_by_one_hot_nd
from PIL import Image
import cv2
import os

inverse_channel_map = {v: k for k, v in channel_map.items()}


def process_segmentation_map(segmentation_map, label):
    if segmentation_map.ndim == 4:
        return list(process_segmentation_map(s, l) for (s, l) in zip(segmentation_map, label))
    assert segmentation_map.ndim == 3
    segmentation_map = segmentation_map.softmax(dim=-3)
    if label == 0:
        segmentation_map[5, :, :] = 0
        segmentation_map[6, :, :] = 0
    elif label == 1:
        segmentation_map[6, :, :] = 0
    elif label == 2:
        segmentation_map[5, :, :] = 0
    segmentation_map = convert_by_one_hot_nd(segmentation_map, nd=2)
    new_segmentation_map = torch.zeros((3, segmentation_map.shape[-2], segmentation_map.shape[-1]), dtype=torch.uint8).to(segmentation_map.device)
    for i in range(len(inverse_channel_map)):
        new_segmentation_map += segmentation_map[i].unsqueeze(0).int() * torch.tensor(inverse_channel_map[i]).unsqueeze(-1).unsqueeze(-1).to(segmentation_map.device)
    return new_segmentation_map.permute(1, 2, 0).cpu().numpy()


def infer(clf, seg_models, loader, device):
    samples = loader.dataset.samples
    for i, x in enumerate(loader):
        x = x.to(device)
        y = clf(x).argmax(-1).item()
        segmentation_map = seg_models[y](x).squeeze(0)
        segmentation_map = process_segmentation_map(segmentation_map, y)
        os.makedirs('result', exist_ok=True)
        filename = "result/" + os.path.splitext(os.path.split(samples[i])[-1])[0] + ".png"
        height, width, _ = cv2.imread(samples[i]).shape
        segmentation_map = cv2.resize(segmentation_map, (width, height), 0, 0, interpolation = cv2.INTER_NEAREST)
        Image.fromarray(segmentation_map).save(
            filename, 
            format="png"
        )
