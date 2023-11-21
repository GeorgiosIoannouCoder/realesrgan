##############################################################################################################
# Filename: unet_discriminator_sn.py
# Description: The archiarchitecture of the UnetDiscriminatorSn used for training the REAL-ESRGAN.
# As descirbed in the paper "Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data"
##############################################################################################################
#
# Import libraries.
#
# Import the MODEL_REGISTRY from the Basicsr library for model registration.
from basicsr.utils.registry import ARCH_REGISTRY

# Import specific functions and classes for deep learning operations from PyTorch.
from torch import nn as nn
from torch.nn import functional as F

# Import spectral_norm to apply spectral normalization to a parameter in the given module.
from torch.nn.utils import spectral_norm


##############################################################################################################
# Register the UnetDiscriminatorSn class in the architecture registry.
@ARCH_REGISTRY.register()
class UnetDiscriminatorSn(nn.Module):
    def __init__(self, num_in_ch, num_feat=64, skip_connection=True):
        super(UnetDiscriminatorSn, self).__init__()
        self.skip_connection = skip_connection
        norm = spectral_norm

        # Define the layers for the U-Net Discriminator with spectral normalization.

        # The first convolution layer.
        self.conv0 = nn.Conv2d(num_in_ch, num_feat, kernel_size=3, stride=1, padding=1)

        # Downsample layers.
        self.conv1 = norm(nn.Conv2d(num_feat, num_feat * 2, 4, 2, 1, bias=False))
        self.conv2 = norm(nn.Conv2d(num_feat * 2, num_feat * 4, 4, 2, 1, bias=False))
        self.conv3 = norm(nn.Conv2d(num_feat * 4, num_feat * 8, 4, 2, 1, bias=False))

        # Upsample layers.
        self.conv4 = norm(nn.Conv2d(num_feat * 8, num_feat * 4, 3, 1, 1, bias=False))
        self.conv5 = norm(nn.Conv2d(num_feat * 4, num_feat * 2, 3, 1, 1, bias=False))
        self.conv6 = norm(nn.Conv2d(num_feat * 2, num_feat, 3, 1, 1, bias=False))

        # Extra convolutional layers.
        self.conv7 = norm(nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=False))
        self.conv8 = norm(nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=False))
        self.conv9 = nn.Conv2d(num_feat, 1, 3, 1, 1)

    def forward(self, x):
        # The first convolution layer.
        x0 = F.leaky_relu(self.conv0(x), negative_slope=0.2, inplace=True)

        # Downsample layers.
        x1 = F.leaky_relu(self.conv1(x0), negative_slope=0.2, inplace=True)
        x2 = F.leaky_relu(self.conv2(x1), negative_slope=0.2, inplace=True)
        x3 = F.leaky_relu(self.conv3(x2), negative_slope=0.2, inplace=True)
        x3 = F.interpolate(x3, scale_factor=2, mode="bilinear", align_corners=False)

        # Upsample layers.
        x4 = F.leaky_relu(self.conv4(x3), negative_slope=0.2, inplace=True)
        if self.skip_connection:  # Skip connections.
            x4 = x4 + x2
        x4 = F.interpolate(x4, scale_factor=2, mode="bilinear", align_corners=False)
        x5 = F.leaky_relu(self.conv5(x4), negative_slope=0.2, inplace=True)
        if self.skip_connection:  # Skip connections.
            x5 = x5 + x1
        x5 = F.interpolate(x5, scale_factor=2, mode="bilinear", align_corners=False)
        x6 = F.leaky_relu(self.conv6(x5), negative_slope=0.2, inplace=True)
        if self.skip_connection:  # Skip connections.
            x6 = x6 + x0

        # Extra convolutional layers.
        out = F.leaky_relu(self.conv7(x6), negative_slope=0.2, inplace=True)
        out = F.leaky_relu(self.conv8(out), negative_slope=0.2, inplace=True)
        out = self.conv9(out)

        return out


##############################################################################################################
