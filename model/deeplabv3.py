def _deeplab_wrapper(_func, num_classes, pretrained):
    from torchvision.models.segmentation.deeplabv3 import DeepLabHead  # noqa
    from torchvision.models.segmentation.fcn import FCNHead  # noqa
    if pretrained:  # Pretrained Model
        net = _func(pretrained=True, progress=False)
        net.classifier = DeepLabHead(2048, num_classes)
        net.aux_classifier = FCNHead(1024, num_classes)
    else:  # Non-pretrained Model
        net = _func(pretrained=False, num_classes=6)
    return net


def deeplabv3_resnet50(num_classes, pretrained):
    from torchvision.models.segmentation import deeplabv3_resnet50
    return _deeplab_wrapper(deeplabv3_resnet50, num_classes, pretrained)


def deeplabv3_resnet101(num_classes, pretrained):
    from torchvision.models.segmentation import deeplabv3_resnet101
    return _deeplab_wrapper(deeplabv3_resnet101, num_classes, pretrained)


__all__ = ['deeplabv3_resnet101', 'deeplabv3_resnet50']
