import segmentation_models_pytorch as smp

def build_model(args):
    decoder = getattr(smp, args["decoder"])
    model = decoder(
        encoder_name=args["encoder"],        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
        in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
        classes=11,                      # model output channels (number of classes in your dataset)
    )
    preprocessing_fn = smp.encoders.get_preprocessing_fn(args["encoder"], "imagenet")
    return model, preprocessing_fn