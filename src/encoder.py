from torch.nn import Module, ModuleList, MaxPool2d
from block import Block
from typing import Tuple, List
from torch import Tensor


class Encoder(Module):
	
	
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