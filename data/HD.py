_base_ = ['blocks']
model = dict(
    strides=(1, ),
    buffer_size=2048,
    max_num_moment=50,
    pyramid_cfg=dict(type="ConvPyramid"),
    pooling_cfg=dict(type="AdaPooling"),
    class_head_cfg=dict(type="ConvHead", kernal_size=3),
    coord_head_cfg=dict(type="ConvHead", kernal_size=3),
    loss_cfg=dict(
        type="BundleLoss",
        loss_cls=dict(type='DynamicBCELoss'),
        loss_reg=None,
        loss_sal=dict(type="SampledNCELoss", direction='row'),
        ),
    )