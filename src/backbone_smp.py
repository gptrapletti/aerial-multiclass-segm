import segmentation_models_pytorch as smp

unet_smp = smp.Unet(
    encoder_name = 'resnet34',
    encoder_depth = 5,
    encoder_weights = None, # 'imagenet'
    decoder_use_batchnorm = True,
    decoder_channels = [256, 128, 64, 32, 16],
    decoder_attention_type = None,
    in_channels = 3,
    classes = 6,
    activation = 'softmax'
)