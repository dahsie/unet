
from encoder import Encoder
from decoder import Decoder
from typing import Tuple
from torch.nn import Module, Conv2d
from torch.nn import functional as F
from torch import Tensor

class UNet(Module):
    """
    UNet model for image segmentation, consisting of an encoder-decoder structure with skip connections. 
    The encoder extracts features from the input, and the decoder reconstructs the segmentation map from these features.

    Parameters:
    -----------
    ouput_dim : Tuple[int]
        The dimensions of the output segmentation map. Typically, this matches the input dimensions to ensure that
        the output has the same spatial dimensions as the input image.
    encoder_channels : Tuple[int], optional
        A tuple specifying the number of channels at each stage of the encoder. The first element corresponds to the
        number of input channels (e.g., 3 for RGB images), and subsequent elements define the number of channels
        in each encoder block. Default is (3, 64, 128, 256, 512, 1024).
    decoder_channels : Tuple[int], optional
        A tuple specifying the number of channels at each stage of the decoder. The first element should match the
        last element of `encoder_channels`, and subsequent elements define the number of channels in each decoder block.
        Default is (1024, 512, 256, 128, 64).
    num_class : int, optional
        The number of output classes for segmentation. For binary segmentation, this should be set to 1. Default is 1.
    retain_dim : bool, optional
        If True, the output segmentation map is resized to match the dimensions specified by `ouput_dim`. Default is True.

    Attributes:
    -----------
    encoder : Encoder
        The encoder part of the UNet, responsible for downsampling and feature extraction.
    decoder : Decoder
        The decoder part of the UNet, responsible for upsampling and reconstructing the segmentation map.
    head : torch.nn.Conv2d
        A 1x1 convolutional layer that reduces the number of channels to `num_class`, producing the final segmentation map.
    retain_dim : bool
        Determines whether to resize the output map to the specified `ouput_dim`.
    ouput_dim : Tuple[int]
        The target dimensions for the output segmentation map, used if `retain_dim` is True.

    Methods:
    --------
    forward(x: torch.Tensor) -> torch.Tensor
        Defines the forward pass of the UNet model. It processes the input tensor through the encoder to extract features,
        then through the decoder to reconstruct the segmentation map. If `retain_dim` is True, the output is resized
        to the specified `ouput_dim`.

    Example:
    --------
    >>> import torch
    >>> from unet import UNet
    >>> model = UNet(ouput_dim=(256, 256), encoder_channels=(3, 64, 128, 256, 512, 1024), decoder_channels=(1024, 512, 256, 128, 64))
    >>> x = torch.randn(1, 3, 256, 256)  # Example input tensor
    >>> output = model(x)
    >>> print(output.shape)  # Expected output shape
    torch.Size([1, 1, 256, 256])
    """
    
    def __init__(self, 
                 ouput_dim: Tuple[int], 
                 encoder_channels: Tuple[int] = (3, 64, 128, 256, 512, 1014),
                 decoder_channels: Tuple[int] = (1014, 512, 256, 128, 64),
                 num_class: int = 1, 
                 retain_dim: bool = True) -> None:
        super(UNet, self).__init__()
        self.encoder = Encoder(channels=encoder_channels)
        self.decoder = Decoder(channels=decoder_channels)
        
        # Initialize the final convolutional layer (regression head) and store class variables
        self.head = Conv2d(in_channels=decoder_channels[-1], out_channels=num_class, kernel_size=1)
        self.retain_dim = retain_dim
        self.ouput_dim = ouput_dim

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through the UNet model.

        Parameters:
        -----------
        x : torch.Tensor
            The input tensor with shape [batch_size, channels, height, width], representing the input image
            or feature map to be segmented.

        Returns:
        --------
        torch.Tensor
            The output tensor representing the segmentation map. The shape is [batch_size, num_class, height, width],
            where `num_class` is the number of output classes. If `retain_dim` is True, the output is resized to match
            the dimensions specified by `ouput_dim`.
        """
        # Extract features from the encoder
        encoder_features_maps = self.encoder(x)
        
        # Decode the features, using skip connections from the encoder
        decoder_features_maps = self.decoder(encoder_features_maps[::-1][0], encoder_features_maps[::-1][1:])
        
        # Apply the final convolution to obtain the segmentation map
        map = self.head(decoder_features_maps)
        
        # If retain_dim is True, resize the output to match ouput_dim
        if self.retain_dim:
            map = F.interpolate(map, size=self.ouput_dim, mode='bilinear', align_corners=False)
        
        return map
