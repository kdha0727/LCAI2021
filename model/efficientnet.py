__all__ = ['get_efficientnet']


def get_efficientnet():
    import torch
    from efficientnet_pytorch import EfficientNet
    from efficientnet_pytorch.utils import GlobalParams
    model = EfficientNet.from_pretrained('efficientnet-b7')
    global_params = model._global_params
    override_global_params = GlobalParams(
        width_coefficient=global_params.width_coefficient,
        depth_coefficient=global_params.depth_coefficient,
        image_size=global_params.image_size,
        dropout_rate=global_params.dropout_rate,
        num_classes=3,
        batch_norm_momentum=global_params.batch_norm_momentum,
        batch_norm_epsilon=global_params.batch_norm_epsilon,
        drop_connect_rate=global_params.drop_connect_rate,
        depth_divisor=global_params.depth_divisor,
        min_depth=global_params.min_depth,
        include_top=True,
    )
    model._global_params = override_global_params
    model._fc = torch.nn.Linear(2560, 3)  # b4: 1792, b7: 2560
