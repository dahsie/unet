from torch.nn import Module, ConvTranspose2d, ModuleList
from torchvision.transforms import CenterCrop
import torch
from torch import Tensor
from block import Block
from typing import Tuple

class Decoder(Module):
    """
    Decoder module for a convolutional neural network, commonly used in architectures like U-Net.
    This module consists of upsampling operations followed by convolutional blocks, reconstructing
    the feature maps from the encoder to produce the final output.

    Parameters:
    -----------
    channels : Tuple[int], optional
        A tuple specifying the number of channels at each stage of the decoder. The first element
        corresponds to the number of input channels for the decoder, and each subsequent element
        defines the number of channels in each upsampling and convolutional block. Default is
        (1014, 512, 256, 128, 64).

    Attributes:
    -----------
    decoder_channels : Tuple[int]
        Stores the number of channels at each stage of the decoder.
    upconvs : torch.nn.ModuleList
        A ModuleList containing `ConvTranspose2d` layers for upsampling. Each layer performs transposed
        convolution to increase the spatial dimensions of the feature maps.
    decoder_blocks : torch.nn.ModuleList
        A ModuleList containing `Block` modules for further processing of the upsampled feature maps.
        Each `Block` applies convolutional operations to refine the feature maps.

    Methods:
    --------
    forward(x: torch.Tensor, encoder_features_maps: Tensor) -> torch.Tensor
        Defines the forward pass through the decoder. The input tensor `x` is upsampled through
        the `ConvTranspose2d` layers and concatenated with the corresponding encoder feature maps.
        The concatenated feature maps are then processed through the decoder blocks to produce the
        final output.
        
    crop(encoder_features_map: Tensor, x: Tensor) -> Tensor
        Crops the encoder feature maps to match the spatial dimensions of the upsampled tensor `x`.
        This ensures that the concatenation in the forward pass aligns the feature maps correctly.

    Example:
    --------
    >>> import torch
    >>> from decoder import Decoder
    >>> decoder = Decoder(channels=(1014, 512, 256, 128, 64))
    >>> x = torch.randn(1, 64, 32, 32)  # Example input tensor with batch size 1, 64 channels, and 32x32 size
    >>> encoder_features = [torch.randn(1, 1014, 8, 8), torch.randn(1, 512, 16, 16), torch.randn(1, 256, 32, 32)]
    >>> output = decoder(x, encoder_features)
    >>> print(output.shape)  # Output shape after passing through the decoder
    torch.Size([1, 64, 32, 32])
    """
    
    def __init__(self, channels: Tuple[int] = (1014, 512, 256, 128, 64)) -> None:
        super(Decoder, self).__init__()
        
        self.decoder_channels = channels
        
        self.upconvs = ModuleList([ConvTranspose2d(channels[i], channels[i + 1], kernel_size=2, stride=2) for i in range(len(channels) - 1)])
        
        self.decoder_blocks = ModuleList([Block(channels[i], channels[i + 1]) for i in range(len(channels) - 1)])
        
    def forward(self, x: torch.Tensor, encoder_features_maps: Tensor) -> torch.Tensor:
        """
        Forward pass through the decoder module.

        Parameters:
        -----------
        x : torch.Tensor
            The input tensor with shape [batch_size, channels, height, width], representing the feature
            maps to be upsampled.
        encoder_features_maps : List[torch.Tensor]
            A list of tensors from the encoder with shapes matching the feature maps at each stage of
            the encoder. These are used for skip connections and concatenation with the upsampled
            features.

        Returns:
        --------
        torch.Tensor
            The output tensor representing the reconstructed feature maps after upsampling and
            applying the decoder blocks. The shape will be [batch_size, output_channels, height, width],
            where `output_channels` is the number of channels at the last stage of the decoder.
        """
        for i in range(len(self.decoder_channels) - 1):
            x = self.upconvs[i](x)  # Apply upsampling to the input tensor
            
            encoder_features_map = self.crop(encoder_features_maps[i], x)
            x = torch.cat([encoder_features_map, x], dim=1)  # Concatenate the cropped encoder features with the upsampled features
            x = self.decoder_blocks[i](x)  # Apply the decoder block to the concatenated features
        
        return x
    
    def crop(self, encoder_features_map: Tensor, x: Tensor) -> Tensor:
        """
        Crop the encoder feature map to match the spatial dimensions of the upsampled tensor `x`.

        Parameters:
        -----------
        encoder_features_map : torch.Tensor
            The encoder feature map tensor with shape [batch_size, channels, height, width] to be cropped.
        x : torch.Tensor
            The upsampled tensor with shape [batch_size, channels, height, width] used to determine the
            target spatial dimensions.

        Returns:
        --------
        torch.Tensor
            The cropped encoder feature map with shape [batch_size, channels, height, width] matching
            the dimensions of `x`.
        """
        (_, _, H, W) = x.shape
        encoder_features_map = CenterCrop([H, W])(encoder_features_map)  # Crop to match dimensions of x
    
        return encoder_features_map
