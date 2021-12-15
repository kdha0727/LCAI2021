import torch

from constants import channel_map
from model.functional import convert_by_one_hot_nd

inverse_channel_map = {v: k for k, v in channel_map.items()}


def process_segmentation_map(segmentation_map, label):
    if segmentation_map.ndim == 4:
        return list(process_segmentation_map(s, l) for (s, l) in zip(segmentation_map, label))
    assert segmentation_map.ndim == 3
    label = label.argmax(dim=-1)
    segmentation_map = segmentation_map.softmax(dim=-3)
    if label == 0:
        segmentation_map[5, :, :] = 0
        segmentation_map[6, :, :] = 0
    segmentation_map = convert_by_one_hot_nd(segmentation_map, nd=2)
    segmentation_map[6, :, :] = 0
    new_segmentation_map = torch.zeros().uint8()
    if label == 2:
        segmentation_map[6, :, :] = segmentation_map[5, :, :]
        segmentation_map[5, :, :] = 0
    for i in range(len(inverse_channel_map)):
        segmentation_map


def get_sensitivity(segmentation_map, original_mask, labels):  # batch unit
    assert segmentation_map.ndim == 4
    segmentation_map = convert_by_one_hot_nd
    for sm, om, la in zip(segmentation_map, original_mask, labels):
        pass
