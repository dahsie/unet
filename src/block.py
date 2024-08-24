
from torch.nn import Module, Conv2d, ReLU
from torch import Tensor


class Block(Module):
	   
	def __init__(self, input_channels: int, output_channels: int) -> None:
		super(Block, self).__init__()

		self.conv1 = Conv2d(in_channels=input_channels, out_channels=output_channels, kernel_size=3)
		self.relu = ReLU()
		self.conv2 = Conv2d(in_channels=output_channels, out_channels=output_channels, kernel_size=3)
		
	def forward(self, x: Tensor) -> Tensor:
		
		return self.conv2(self.relu(self.conv1(x)))