from torch.nn import Module, ConvTranspose2d, ModuleList
from torchvision.transforms import CenterCrop
import torch
from torch import Tensor
from block import Block
from typing import Tuple

class Decoder(Module):
	
	def __init__(self, channels: Tuple[int]=(1014, 512, 256, 128, 64)) -> None:
		
		super(Decoder, self).__init__()
		
		self.decoder_channels = channels = channels
		
		self.upconvs = ModuleList([ConvTranspose2d(channels[i], channels[i + 1], 2, 2) for i in range(len(channels) - 1)])
		
		self.decoder_blocks = ModuleList([Block(channels[i], channels[i + 1]) for i in range(len(channels) - 1)])
		

	def forward(self, x: torch.Tensor, encoder_features_maps: Tensor) -> Tensor:
		
		for i in range(len(self.decoder_channels) - 1):
			
			x = self.upconvs[i](x) # pass the inputs through the upsampler blocks
			
			encoder_features_map = self.crop(encoder_features_maps[i], x)
			x = torch.cat([encoder_features_map, x], dim=1)
			x = self.decoder_blocks[i](x)
		
		return x
	
	def crop(self, encoder_features_map: Tensor, x: Tensor) -> Tensor:
		
		
		(_, _, H, W) = x.shape
		encoder_features_map = CenterCrop([H, W])(encoder_features_map) #Crop the encoder feature map to match the dimensions of the input tensor `x`
	
		return encoder_features_map