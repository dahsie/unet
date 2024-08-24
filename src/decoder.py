from torch.nn import Module, ConvTranspose2d, ModuleList
from torchvision.transforms import CenterCrop
import torch
from torch import Tensor
from block import Block
from typing import Tuple

class Decoder(Module):
	"""
    The Decoder class reconstructs the spatial resolution of feature maps
    using a series of upsampling and convolutional blocks. It is commonly
    used in decoder architectures of convolutional neural networks, such as U-Net.

    Parameters:
    -----------
    channels : Tuple[int], optional
        A tuple defining the number of channels for each decoder block. The first
        element is the number of channels of the input tensor, and subsequent elements
        correspond to the channels in the decoder blocks. Default is (1024, 512, 256, 128, 64).

    Attributes:
    -----------
    decoder_channels : Tuple[int]
        Stores the number of channels for each stage of the decoder.
    upconvs : torch.nn.ModuleList
        A ModuleList containing ConvTranspose2d layers for upsampling the input feature maps.
    decoder_blocks : torch.nn.ModuleList
        A ModuleList containing `Block` modules that apply convolutions after upsampling and concatenation.

    Methods:
    --------
    forward(x: torch.Tensor, encoder_features_map: Tensor) -> Tensor
        Passes the input tensor through the decoder, upsampling and concatenating it with 
        corresponding feature maps from the encoder before applying the convolutional blocks.

    crop(encoder_features_map: Tensor, x: Tensor) -> Tensor
        Crops the encoder feature map to match the spatial dimensions of the input tensor `x`.

    Example:
    --------
    >>> decoder = Decoder(channels=(1024, 512, 256, 128, 64))
    >>> x = torch.randn(1, 1024, 16, 16)  # Example input tensor
    >>> encoder_features_map = [torch.randn(1, 512, 32, 32), torch.randn(1, 256, 64, 64),
    >>>                         torch.randn(1, 128, 128, 128), torch.randn(1, 64, 256, 256)]
    >>> output = decoder(x, encoder_features_map)
    >>> print(output.shape)  # Expected output shape after decoding
    torch.Size([1, 64, 256, 256])
    """
	def __init__(self, channels: Tuple[int]=(1014, 512, 256, 128, 64)) -> None:
		
		super(Decoder, self).__init__()
		
		self.decoder_channels = channels = channels
		
		self.upconvs = ModuleList([ConvTranspose2d(channels[i], channels[i + 1], 2, 2) for i in range(len(channels) - 1)])
		
		self.decoder_blocks = ModuleList([Block(channels[i], channels[i + 1]) for i in range(len(channels) - 1)])
		

	def forward(self, x: torch.Tensor, encoder_features_maps: Tensor) -> Tensor:
		"""
        Forward pass of the decoder. 

        Parameters:
        -----------
        x : Tensor
            The input tensor to be upsampled and decoded, typically the output of the previous decoder block or the bottleneck.
        encoder_features_map : Tensor
            A list of feature maps from the encoder that will be concatenated with upsampled features in the decoder.

        Returns:
        --------
        Tensor
            The decoded tensor after passing through all upsampling and convolutional blocks.
        """
		for i in range(len(self.decoder_channels) - 1):
			
			x = self.upconvs[i](x) # pass the inputs through the upsampler blocks
			
			encoder_features_map = self.crop(encoder_features_maps[i], x)
			x = torch.cat([encoder_features_map, x], dim=1)
			x = self.decoder_blocks[i](x)
		
		return x
	
	def crop(self, encoder_features_map: Tensor, x: Tensor) -> Tensor:
		"""
        Crop the encoder feature map to match the dimensions of the input tensor `x`.

        Parameters:
        -----------
        encoder_features_map : Tensor
            The feature map from the encoder to be cropped.
        x : Tensor
            The input tensor with the desired spatial dimensions.

        Returns:
        --------
        Tensor
            The cropped encoder feature map with the same height and width as `x`.
        """
		
		(_, _, H, W) = x.shape
		encoder_features_map = CenterCrop([H, W])(encoder_features_map)
	
		return encoder_features_map