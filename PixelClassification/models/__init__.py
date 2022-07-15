from PixelClassification.models.UNet import UNet
from PixelClassification.models.UNet3D import UNet3D

def get_model(name, model_opts):
    if name == "unet":
        model = UNet(**model_opts)
        return model
    elif name == "unet3d":
        model = UNet3D(**model_opts)
        return model
    else:
        raise RuntimeError("model \"{}\" not available".format(name))
