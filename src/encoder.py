from torch.nn import Module, ModuleList, MaxPool2d
from block import Block
from typing import Tuple, List
from torch import Tensor


class Encoder(Module):
	"""
    The Encoder class constructs a series of convolutional blocks followed by max pooling layers,

    Parameters:
    -----------
    channels : tuple of int, optional
        A tuple defining the number of channels for each convolutional block.
        The length of the tuple determines the number of encoder blocks.
        Default is (3, 16, 32, 64), where:
            - 3 is the number of input channels (e.g., RGB images).
            - 16, 32, 64 are the number of output channels for subsequent blocks.

    Attributes:
    -----------
    encBlocks : torch.nn.ModuleList
        A ModuleList containing sequential `Block` modules that perform convolution operations.
    pool : torch.nn.MaxPool2d
        A MaxPool2d layer with a kernel size of 2, applied after each convolutional block to reduce spatial dimensions.

    Methods:
    --------
    forward(x)
        Passes the input tensor through each encoder block followed by max pooling.
        Returns a list of outputs from each block before pooling, which can be useful for skip connections in decoder parts.

    Example:
    --------
    >>> import torch
    >>> from encoder import Encoder
    >>> encoder = Encoder(channels=(3, 64, 128, 256, 512, 1024))
    >>> x = torch.randn(1, 3, 572, 572)  # Batch size of 1, 3 input channels, 256x256 image
    >>> features = encoder(x)
    >>> for f in features:
    ...     print(f.shape)
    torch.Size([1, 64, 568, 568])
    torch.Size([1, 128, 280, 280])
    torch.Size([1, 256, 136, 136])
    torch.Size([1, 512, 64, 64])
    torch.Size([1, 1024, 28, 28])
    """
	
	def __init__(self, channels: Tuple[int]=(3, 64, 128, 256, 512, 1014))-> None:
		super(Encoder, self).__init__()
		# store the encoder blocks and maxpooling layer
		self.encoder_channels = channels
		self.encoder_blocks = ModuleList([Block(input_channels=channels[i], output_channels=channels[i + 1]) for i in range(len(channels) - 1)])
		self.pool = MaxPool2d(2)
		
	def forward(self, x: Tensor) ->List[Tensor]:
		# initialize an empty list to store the intermediate outputs
		blockOutputs: List[Tensor] = []
		# loop through the encoder blocks
		for block in self.encoder_blocks:
			# pass the inputs through the current encoder block, store
			# the outputs, and then apply maxpooling on the output
			x = block(x)
			blockOutputs.append(x)
			x = self.pool(x)
		# return the list containing the intermediate outputs
		return blockOutputs