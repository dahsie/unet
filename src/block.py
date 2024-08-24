
from torch.nn import Module, Conv2d, ReLU
from torch import Tensor


class Block(Module):
	"""
    A basic convolutional neural network block consisting of two convolutional layers and one ReLU activation layer.

    This block applies a convolution using `conv1` followed by a ReLU activation, 
    and then applies a second convolution using `conv2`. The output is the result of 
    this second convolution.

    Attributes:
    -----------
    conv1 : torch.nn.Conv2d
        The first convolutional layer with a 3x3 kernel size.
    relu : torch.nn.ReLU
        The ReLU activation function applied after the first convolution.
    conv2 : torch.nn.Conv2d
        The second convolutional layer with a 3x3 kernel size.

    Parameters:
    -----------
    input_channels : int
        The number of input channels for the first convolutional layer.
    output_channels : int
        The number of output channels for both convolutional layers.
        
    Methods:
    --------
    forward(x)
        Defines the forward pass of the block. Takes input tensor `x` and returns the output tensor 
        after applying conv1, ReLU, and conv2 sequentially.
    
    Example:
    --------
    >>> block = Block(input_channels=3, output_channels=64)
    >>> x = torch.randn(1, 3, 224, 224)  # Example input tensor
    >>> output = block(x)
    """
	def __init__(self, input_channels: int, output_channels: int) -> None:
		super(Block, self).__init__()

		self.conv1 = Conv2d(in_channels=input_channels, out_channels=output_channels, kernel_size=3)
		self.relu = ReLU()
		self.conv2 = Conv2d(in_channels=output_channels, out_channels=output_channels, kernel_size=3)
		
	def forward(self, x: Tensor) -> Tensor:
		
		return self.conv2(self.relu(self.conv1(x)))