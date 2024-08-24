from torch.nn import Module, ModuleList, MaxPool2d
from block import Block
from typing import Tuple, List
from torch import Tensor


class Encoder(Module):
    """
    Encoder module for a convolutional neural network, typically used in architectures like U-Net.
    This module consists of a series of convolutional blocks followed by max pooling layers to 
    progressively downsample the input and extract features.

    Parameters:
    -----------
    channels : Tuple[int], optional
        A tuple specifying the number of channels at each stage of the encoder. The first element
        corresponds to the number of input channels, and each subsequent element specifies the number
        of channels in each convolutional block. Default is (3, 64, 128, 256, 512, 1024).

    Attributes:
    -----------
    encoder_channels : Tuple[int]
        Stores the number of channels at each stage of the encoder.
    encoder_blocks : torch.nn.ModuleList
        A ModuleList containing `Block` modules, each applying convolutional operations with ReLU activation.
    pool : torch.nn.MaxPool2d
        A max pooling layer with a 2x2 kernel used to downsample the feature maps after each encoder block.

    Methods:
    --------
    forward(x: torch.Tensor) -> List[torch.Tensor]
        Defines the forward pass through the encoder. The input tensor `x` is passed through each
        convolutional block, followed by max pooling, and intermediate outputs are collected into a list.
        
    Example:
    --------
    >>> import torch
    >>> from encoder import Encoder
    >>> encoder = Encoder(channels=(3, 64, 128, 256, 512, 1024))
    >>> x = torch.randn(1, 3, 256, 256)  # Example input tensor with batch size 1, 3 channels, and 256x256 size
    >>> features = encoder(x)
    >>> for f in features:
    ...     print(f.shape)
    torch.Size([1, 64, 128, 128])
    torch.Size([1, 128, 64, 64])
    torch.Size([1, 256, 32, 32])
    torch.Size([1, 512, 16, 16])
    torch.Size([1, 1024, 8, 8])
    """
    
    def __init__(self, channels: Tuple[int] = (3, 64, 128, 256, 512, 1024)) -> None:
        super(Encoder, self).__init__()
        self.encoder_channels = channels
        self.encoder_blocks = ModuleList([Block(input_channels=channels[i], output_channels=channels[i + 1]) for i in range(len(channels) - 1)])
        self.pool = MaxPool2d(2)
        
    def forward(self, x: Tensor) -> List[Tensor]:
        """
        Forward pass through the encoder module.

        Parameters:
        -----------
        x : torch.Tensor
            The input tensor with shape [batch_size, channels, height, width], where `channels` is
            the number of input channels.

        Returns:
        --------
        List[torch.Tensor]
            A list of tensors representing the output of each encoder block after applying 
            convolution and max pooling. Each tensor in the list has the shape [batch_size, output_channels, height/2, width/2],
            reflecting the downsampling at each stage.
        """
        blockOutputs: List[Tensor] = []
        for block in self.encoder_blocks:
            x = block(x)
            blockOutputs.append(x)
            x = self.pool(x)
        return blockOutputs
