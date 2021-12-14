import torch
import torch.nn.functional as _F  # noqa

_default_reduction = 'mean'
_epsilon = 1e-7


# Losses


def _apply_reduction(tensor, reduction):
    if reduction is None:
        return tensor
    elif reduction == 'mean':
        return tensor.mean()
    elif reduction == 'sum':
        return tensor.sum()
    raise ValueError("Reduction expected to be None, 'mean', or 'sum', got '%s'" % reduction)


def _dice_loss_2d(mul, add, reduction=_default_reduction, epsilon=_epsilon):  # for multiclass
    mul, add = mul[:, :-1, :, :], add[:, :-1, :, :]
    intersection = mul.sum(dim=(-2, -1)) + epsilon
    union = add.sum(dim=(-2, -1)) + epsilon * 2
    loss = 1. - (2. * intersection / union)
    return _apply_reduction(loss, reduction)


def _iou_loss_2d(mul, add, reduction=_default_reduction, epsilon=_epsilon):  # for multiclass
    mul, add = mul[:, :-1, :, :], add[:, :-1, :, :]
    intersection = mul.sum(dim=(-2, -1)) + epsilon
    union = (add - mul).sum(dim=(-2, -1)) + epsilon
    loss = 1. - (intersection / union)
    return _apply_reduction(loss, reduction)


def dice_loss_2d(output, target, reduction=_default_reduction):
    return _dice_loss_2d(output * target, output + target, reduction=reduction)


def iou_loss_2d(output, target, reduction=_default_reduction):
    return _iou_loss_2d(output * target, output + target, reduction=reduction)


# Utils


def convert_by_one_hot_nd(tensor, nd):
    index = tensor.argmax(dim=-nd - 1).long()
    return torch.zeros_like(tensor).scatter(
        dim=-nd - 1, index=index.unsqueeze(dim=-nd - 1), src=torch.ones_like(tensor)).to(tensor.dtype)


def one_hot_nd(tensor, n_classes, nd):  # N H W
    new_shape = list(range(tensor.ndim))
    new_shape.insert(-nd, tensor.ndim)
    return _F.one_hot(tensor.long(), n_classes).permute(new_shape)  # N C H W


def __getattr__(name):
    from torch.nn import functional
    return getattr(functional, name)
