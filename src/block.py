
from torch.nn import Module, Conv2d, ReLU, BatchNorm2d
from torch import Tensor


class Block(Module):
    """
    A basic building block for a convolutional neural network, consisting of two convolutional
    layers with ReLU activation in between. This block is commonly used in encoder-decoder
    architectures like U-Net.

    Parameters:
    -----------
    input_channels : int
        The number of input channels for the first convolutional layer.
    output_channels : int
        The number of output channels for both convolutional layers.

    Attributes:
    -----------
    conv1 : torch.nn.Conv2d
        The first convolutional layer with a 3x3 kernel, which reduces the spatial dimensions 
        and increases the depth to `output_channels`.
    relu : torch.nn.ReLU
        The ReLU activation function applied after the first convolutional layer.
    conv2 : torch.nn.Conv2d
        The second convolutional layer with a 3x3 kernel, which further processes the feature
        maps produced by the first convolutional layer.

    Methods:
    --------
    forward(x: torch.Tensor) -> torch.Tensor
        Defines the forward pass of the block. The input tensor `x` is passed through the first 
        convolutional layer, followed by ReLU activation, and then through the second convolutional layer.
        
    Example:
    --------
    >>> import torch
    >>> block = Block(input_channels=3, output_channels=64)
    >>> x = torch.randn(1, 3, 256, 256)  # Example input tensor with batch size 1, 3 channels, and 256x256 size
    >>> output = block(x)
    >>> print(output.shape)  # Output shape after passing through the block
    torch.Size([1, 64, 254, 254])
    """
    
    def __init__(self, input_channels: int, output_channels: int) -> None:
        super(Block, self).__init__()

        self.conv1 = Conv2d(in_channels=input_channels, out_channels=output_channels, kernel_size=3)
        self.bn1 = BatchNorm2d(output_channels)
        self.relu1 = ReLU()
        self.conv2 = Conv2d(in_channels=output_channels, out_channels=output_channels, kernel_size=3)
        self.bn2 = BatchNorm2d(output_channels)
        self.relu2 = ReLU()

        
    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through the Block module.

        Parameters:
        -----------
        x : torch.Tensor
            The input tensor with shape [batch_size, input_channels, height, width].

        Returns:
        --------
        torch.Tensor
            The output tensor after passing through two convolutional layers and a ReLU activation.
            The shape will be [batch_size, output_channels, height-2, width-2] due to the 3x3 convolutions.
        """
        # return self.bn2(self.conv2(self.relu(self.bn1(self.conv1(x)))))
        # return self.relu2(self.conv2(self.relu1(self.conv1(x))))
        return self.relu2(self.bn2(self.conv2(self.relu1(self.bn1(self.conv1(x))))))
