
from encoder import Encoder
from decoder import Decoder
from typing import Tuple
from torch.nn import Module, Conv2d
from torch.nn import functional as F

class UNet(Module):
	"""HeLFHFS;Bjf
	FQSN;NFSHSQ
	qs;,FS;QJS;j
	"""
	
	def __init__(self,ouput_dim : Tuple[int], 
			  encoder_channels: Tuple[int]=(3, 64, 128, 256, 512, 1014),
			  decoder_channels: Tuple[int]=(1014, 512, 256, 128, 64),
			  num_class=1, retain_dim=True)-> None:
		
		super(UNet, self).__init__()
		self.encoder = Encoder(channels=encoder_channels)
		self.decoder = Decoder(channels=decoder_channels)
		
		# Initialize the final convolutional layer (regression head) and store class variables
		self.head = Conv2d(in_channels=decoder_channels[-1], out_channels=num_class, kernel_size=1)
		self.retain_dim = retain_dim
		self.ouput_dim = ouput_dim
		
        
	def forward(self, x):
		"""
        Forward pass of the UNet model.

        Parameters:
        -----------
        x : torch.Tensor
            The input tensor, typically a batch of images with dimensions [batch_size, channels, height, width].

        Returns:
        --------
        torch.Tensor
            The output tensor representing the segmentation map. The dimensions of the output
            will match `ouput_dim` if `retainDim` is set to True.
        """
		
        # Extract features from the encoder
		encoder_features_maps = self.encoder(x)
		
		# Decode the features, using skip connections from the encoder
		decoder_features_maps = self.decoder(encoder_features_maps[::-1][0],encoder_features_maps[::-1][1:])
		
        # Apply the final convolution to obtain the segmentation map
		map = self.head(decoder_features_maps)
		
        # If retain_dim is True, resize the output to match ouput_dim
		if self.retain_dim:
			map = F.interpolate(map, self.ouput_dim)
		
		return map