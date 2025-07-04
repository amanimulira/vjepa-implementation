from dataclasses import dataclass
from typing import Callable, Optional, Union

import torch
from torch import nn

from ...activations import ACT2FN
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import BaseModelOutput, ImageClassifierOutput
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from ...utils import ModelOutput, auto_docstring, can_return_tuple, logging
from .configuration_vjepa2 import VJEPA2Config

logger = logging.get_logger(__name__)

"""
These dataclasses are crucical for organizing and returning the various outputs generated 
by the VJEPA2 model, especially when dealing with masked inputs and a separte predictor 
component. Providing a structured way to access differnet types of information, i.e. 
hidden states at various layers and attention weights.

"""

# Class is designed to hold the outputs from the VJEPA Predictor. 
@dataclass
@auto_docstring(
	custom_intro="""
	VJEPA Predictor outputs that also contains the masked encoder outputs
	"""
)
class VJEPAWithMasksedInputPredictorOutput(ModelOutput):
	r"""
	masked_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*, returned when `context_mask` is provided which is applied on VJEPAEncoder outputs):
		The masked hidden state of the model.
	target_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*, returned when `target_mask` is provided which is applied on VJEPA2Encoder outputs):
		The target hidden state of the model.
	"""

	last_hidden_state: torch.FloatTensor
	masked_hidden_state: Optional[torch.FloatTensor] = None
	hidden_states: Optional[tuple[torch.FlaotTensor, ...]] = None
	attentions: Optional[tuple[torch.FloatTensor, ...]] = None
	target_hidden_state: Optional[torch.FloatTensor] = None


# This class encapsulates the outputs of the entire VJEPA model, including the masked encoder outputs and optionally the predictor outputs.
@dataclass
@auto_docstring(
	custom_intro="""
	VJEPA outputs that also contains the masked encoder outputs
	Optionally contains the predictor outputs
	"""
)
class VJEPAWithMaskedInputModelOutput(ModelOutput):
	r"""
	masked_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*, returned when `context_mask` is provided which is applied on VJEPA2Encoder outputs):
		The masked hidden state of the model.
	predictor_output (`VJEPA2WithMaskedInputPredictorOutput`, *optional*):
		The output form the Predictor module.
	"""

	last_hidden_state: torch.FloatTensor
	masked_hidden_state: Optional[torch.FloatTensor] = None
	hidden_states: Optional[tuple[torch.FloatTensor, ...]] = None
	attentions: Optional[tuple[torch.FloatTensor, ...]] = None
	predictor_output: Optional[VJEPA2WithMaskedInputPredictorOutput] = None

	def to_tuple(self):
		output = list(super().to_tuple())
		if isinstance(output[-1], VJEPA2WithMaskedInputPredictorOutput):
			output[-1] = output[-1].to_tuple()
		return tuple(output)

# Converts raw 3D video data into a sequence of embeddings that a transformer model can process.

class VJEPA2PatchEmbeddings3D(nn.Module):
	"""
	Image to Patch Embedding
	"""

	def __init__(
			self, 
			config: VJEPA2Config,
			hidden_size: int = 1024,
	):
		super().__init__()
		# Stores the size of 2D patches that images will be divided into
		self.patch_size = config.patch_size 
		# Stores the size of tubelets in the temporal dimension.
		# How many consecutive frames are grouped together to form a 3D patch
		self.tubelet_size = config.tubelet_size 
		# Stores the dimension of the output embeddings
		self.hidden_size = hidden_size

		# Core of patch embedding.
		self.proj = nn.Conv3d(
			in_chanenls=config.in_chans,
			out_channels=hidden_size,
			kernel_size=(config.tubelet_size, config.patch_size, config.patch_size),
			stride=(config.tubelet_size, config.patch_size, config.patch_size)
		)
	# Calculates the total number of patches the will be generated: form given video input.
	@staticmethod
	def num_patches(config):
		""" 
		Divides frames_per_clip by tubelet_size -> temporal segments
		Divides crop_size by patch_size -> number of 2D patches
		"""
		return (
			(config.frames_per_clip // config.tubelet_size)
			* (config.crop_size // config.patch_size)
			* (config.crop_size // config.patch_size)
		)
	
	def forward(self, pixel_values_videos: torch.Tensor) -> torch.Tensor:
		"""
		Takes pixel_values_videos, torch.Tensor, representing raw video data.

		Apply 3D convolution to input video.

		(batch_size, hidden_size, num_temporal_patches, num_height_patches, num_width_patches)

		.flatten(2) flattens dimensions from the 3rd dim onwards
		(num_temporal_patches, num_height_patches, num_width_patches)
		into a single dim.

		spatial + temporal patches combined into a single sequence.

		(batch_size, hidden_size, num_total_patches).

		.transpose(1, 2) swaps 2nd and 3rd dimensions

		(batch_size, hidden_size, num_total_patches)
		"""
		x = self.proj(pixel_values_videos).flatten(2).transpose(1, 2)
		return x
	
"""

Takes raw video pixel values and transforming them into the initial embeddings that are then 
fed into the VJEPA2 encoder. Orchestrates the patch embeddings process and handles potential
input inconsistencies.

Basically the initial preprocessing step for video input.

	- ensures its in the correct format 
	- has enough frames for 3D patch embedding operation to succeed

Then pass it on as feature embeddings.

"""
class VJEPA2Embeddings(nn.Module):
	"""
	Construct mask token, position and patch embeddings
	"""

	def __init__(self, config: VJEPA2Config, hidden_size: int = 1024):
		super().__init__()

		self.config = config
		self.hidden_size = hidden_size
		# CORE COMPONENT: Converts raw video into patch embeddings.
		self.patch_embeddings = VJEPA2PatchEmbeddings3D(config, hidden_size=hidden_size)

		self.num_patches = self.patch_embeddings.num_patches
		self.patch_size = config.patch_size

	# forward pass for generating embeddings
	def forward(self, pixel_values_videos: torch.Tensor) -> torch.Tensor:
		num_frames = pixel_values_videos.shape[1]

		# Swap 'frames' and 'channels' dims, the result is:
		# (batch_size, channels, num_frames, height, width)
		pixel_values_videos = pixel_values_videos.permute(0, 2, 1, 3, 4)

		# If the input vision (image/video) consists of num_frames < tubelet_size,
		# then embedding lookup fails; we duplicate the frames.
		if num_frames < self.config.tubelet_size:
			pixel_values_videos = pixel_values_videos.repeat(1, 1, self.config.tubelet_size, 1, 1)
		
		# Ensures data type consistency.
		target_dtype = self.patch_embeddings.proj.weight.dtype
		# Cast input to same data type as convolutional layer's weights
		# Why? Good for numerical stability and compatibility ( especially when using mixed precision training )
		pixel_values_videos = pixel_values_videos.to(dtype=target_dtype)
		# Performs 3D convolution and reshaping to get patch embeddings
		embeddings = self.patch_embeddings(pixel_values_videos)

		return embeddings
	


