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
	
"""

Utility function for performing multi-head self-attention in an "eager" fashion.

"""
def eager_attention_forward(
		module: nn.Module,
		query: torch.Tensor,
		key: torch.Tensor,
		value: torch.Tensor,
		attention_mask: Optional[torch.Tensor],
		scaling: float,
		dropout: float = 0.0,
		**kwargs,
):
	# Take the dot product between "query" and "key" to get the raw attention scores.
	# Transpose last 2 dims; perform batch matrix multiplication between query and transposed key.
	# Result represents raw attention scores for each head, where matrix indicates how much each
	# token in the query attends to each token in the key. Then scale.
	attn_weights = torch.matmul(query, key.transpose(-1, -2)) * scaling

	# Normalize the attention scores to probs.
	# Applies the softmax function along the last dimension. Normalizing the score so they sum to 1.
	attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)

	# This is actually dropping out entire tokens to attend to, which might
	# seem a bit unusal, byt is taken from the original Transformer paper.
	attn_weights = nn.functional.dropout(attn_weights, p=dropout, trainig=module.training)

	# Mask heads if we want...
	# Applied to the attention weights: where prevents attention to those positions.
	if attention_mask is not None:
		attn_weights = attn_weights * attention_mask

	attn_output = torch.matmul(attn_weights, value)
	# Ensures memory layout of the tensor is contiguous after transpose operation
	attn_output = attn_output.transpose(1, 2).contiguous()

	return attn_output, attn_weights


# RoPE type of positional encoding that encodes relative positional information into the self-attention mechanism
# How? By rotating the query and key vectors.

"""

Takes input, typically query or key vectors from attention layer.

Tensor represents positional indices for each token.

"""
def rotate_queries_or_keys(x, pos):
	# Extracts batch size, number of attention heads, sequence length and head dimension from the tensor x.
	B, num_heads, N, D = x.size()

	"""
	like inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
	computed every time. instead HF style is to compute the inv_freq once and store it
	-- compute angle for each position.
	"""

	# Calculate Angualar Frequencies
	omega = torch.arange(D // 2, dtype=x.dtype, device=x.device)
	omega /= D / 2.0
	omega = 1.0 / 10000**omega # (D/2, )
	# Outer product between the pos, positional indices and omega, angular frequencies
	# Einstein summation operation
	freq = torch.einsum("..., f -> ... f", pos, omega) # (..., N, D/2), outer product

	# -- build rotation matrix and apply
	emb_sin = freq.sin() # (..., N, D/2)
	emb_cos = freq.cos() # (..., N, D/2)

	# Duplicates the last dim, as RoPE applies pairs of dim
	# i.e. (x0, x1) rotated by theta -> (x0*cos(theta) - x1*sin(theta), x0*sin(theta) + x1*cos(theta))
	emb_sin = emb_sin.squeeze(-1).repeat(1, 1, 1, 2)
	emb_cos = emb_cos.squeeze(-1).repeat(1, 1, 1, 2)
	"""

	1. x.unflatten(-1, (-1, 2)): reshapes last dim D of x into (D/2, 2). Grouping elements into pairs

	2. y.unbind(dim=-1): splits (D/2, 2) dim into two separate tensors, y1 (containing x0, x2, ...)
	and y2 (containing x1, x3, ...)

	3. torch.stack((-y2, y1), dim=-1): creates rotated components. for pair (a, b), the rotated pair 
	becomes (a*cos - b*sin, a*sin + b*cos). y here represents the (-b, a) part of the rotation.

	4. y.flatten(-2): Flattens the (D/2, 2) back to D.

	5. returns (x * emb_cos) + (y * emb_sin): final rotation formula.

	[ cos(theta)  -sin(theta) ] [ x0 ]   = [ x0*cos(theta) - x1*sin(theta) ]
	[ sin(theta)   cos(theta) ] [ x1 ]     [ x0*sin(theta) + x1*cos(theta) ]

	"""
	y = x.unflatten(-1, (-1, 2))
	y1, y2 =y.unbid(dim=-1)

	y = torch.stack((-y2, y1), dim=-1)
	y = y.flatten(-2)

	return (x * emb_cos) + (y * emb_sin)

"""

Implements self-attention mechanism that incorporates Rotary Positional Embeddings (RoPE). Special attention
layer for the VJEPA2 mdoel, designed for video data.

"""

class VJEPA2RopeAttention(nn.Module):
	def __init__(
			self, 
			config: VJEPA2Config,
			hidden_size: int = 1024,
			num_attention_heads: int = 16,
	):
		super().__init__()
		self.config = config
		self.hidden_size = hidden_size
		self.num_attention_heads = num_attention_heads

		if hidden_size % num_attention_heads != 0:
			raise ValueError(
				f"The hidden size {(hidden_size)} is not a multiple of the number of attention "
				f"heads {num_attention_heads}"
			)
		
		self.attention_head_size = int(hidden_size / num_attention_heads)
		self.all_head_size = self.num_attention_heads * self.attention_head_size
		# Linear Projection: Projects input, hidden_size to all_head_size for queries, keys, and values
		self.query = nn.Linear(hidden_size, self.all_head_size, bias=config.qkv_bias)
		self.key = nn.Linear(hidden_size, self.all_head_size, bias=config.qkv_bias)
		self.value = nn.Linear(hidden_size, self.all_head_size, bias=config.qkv_bias)

		self.proj = nn.Linear(hidden_size, hidden_size)
		self.dropout_prob = config.attention_probs_dropout_prob
		self.dropout = nn.Dropout(self.dropout_prob)
		# Number of patches along height/width.
		self.grid_size = self.config.crop_size // self.config.patch_size
		# Number of tubelets alogn temporal dimension.
		self.grid_depth = self.config.frames_per_clip // self.config.tubelet_size
		"""
		How attention_head_size is split to apply RoPE independently across

			1. Depth
			2. Height
			3. Width

		roughly 1/3 of the head dim
		"""
		self.d_dim = int(2 * ((self.attention_head_size // 3) // 2))
		self.h_dim = int(2 * ((self.attention_head_size // 3) // 2))
		self.w_dim = int(2 * ((self.attention_head_size // 3) // 2))

		self.scaling = self.attention_head_size**-0.5
		self.is_causal = False
	
	# Reshape the query, key or value tensors from (batch_size, sequence_lenth, all_head_size) to
	# (batch_size, num_attention_heads, sequence_length, attention_head_size)
	def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
		new_x_shape = x.size()[:-1] + (
			self.num_attention_heads,
			self.attention_head_size,
		)
		x = x.view(new_x_shape)
		return x.permute(0, 2, 1, 3)
	
	# Extract frame (depth) and height components of a given 1D token ID, assuming a flattened 3D grid of patches.
	# Reverse the flattening process to get the 3D coordinates
	def _get_frame_pos(self, ids):
		tokens_per_frame = int(self.grid_size * self.grid_size)
		return ids // tokens_per_frame
	
	def _get_height_pos(self, ids):
		# Remove frame component form ids
		tokens_per_frame = int(self.grid_size * self.grid_size)
		frame_ids = self._get_frame_pos(ids)
		ids = ids - tokens_per_frame * frame_ids

		tokens_per_row = self.grid_size
		return ids // tokens_per_row
	
	# Calculates the 3D positional IDs (frame, height, width) for each token.
	# Assumes sequential arrangement.
	# Decomposes the 1D token index into its 3D coordinates based on the grid_size and 
	# tokens_per_frame. Essential for applying 3D RoPE
	def get_position_ids(self, x, masks=None):
		device = x.device
		token_size = x.size(1)

		# Note: when masks is none, we use a 1d id instead of Bxnum_attention_heads mask,
		# as 1d vector is breadcasted to the correct shapes.

		if masks is not None:
			ids = masks.unsqueeze(1).repeat(1, self.num_attention_heads, 1)
		else:
			ids = torch.arange(token_size, device=device)
		
		# Change to allow for extrapolation
		tokens_per_frame = int(self.grid_size * self.grid_size)
		frame_ids = self._get_frame_pos(ids)

		tokens_per_row = self.grid_size
		height_ids = self._get_height_pos(ids)

		# Remove frame component from ids (1st term) and height component (2nd term)
		width_ids = (ids - tokens_per_frame * frame_ids) - tokens_per_row * height_ids
		return frame_ids, height_ids, width_ids
	"""
	Applies rotate_queries_or_keys function to different parts of the query/key vector
	"""
	def apply_rotary_embeddings(self, qk, pos_ids):
		# Unpacks the 3D positional IDs.
		d_mask, h_mask, w_mask = pos_ids
		s = 0

		# Slices the qk tensor based on self.d_dim, self.h_dim, self.w_dim and applies rotate_queries_or_keys to each slice
		qkd = rotate_queries_or_keys(qk[..., s : s + self.d_dim], pos=d_mask)
		s += self.d_dim
		qkh = rotate_queries_or_keys(qk[..., s : s + self.h_dim], pos=h_mask)
		s += self.h_dim
		qkw = rotate_queries_or_keys(qk[..., s : s + self.w_dim], pos=w_mask)
		s += self.w_dim

		# combine rotated dimension
		if s < self.attention_head_size:
			qkr = qk[..., s:]
			qk = torch.cat([qkd, qkh, qkw, qkr], dim=-1)
		else:
			qk = torch.cat([qkd, qkh, qkw], dim=-1)
		return qk

	"""
	mixed_query_layer = self.query(hidden_states): Computes raw queries.

	Computes and reshapes keys.

	key_layer = self.transpose_for_scores(self.key(hidden_states)) 
	value_layer = self.transpose_for_scores(self.value(hidden_states))
	query_layer = self.transpose_for_scores(mixed_query_layer)

	
	Apply RoPE.

	pos_ids = self.get_position_ids(hidden_states, masks=position_mask): Gets the 3D positional IDs
	key_layer = self.apply_rotary_embeddings(key_layer, pos_ids): Applies RoPE to the key vectors
	query_layer = self.apply_rotary_embeddings(query_layer, pos_ids): Applies RoPE to the query vectors
	"""
	def forward(
			self, 
			hidden_states,
			position_mask: Optional[torch.Tensor] = None,
			output_attentions: bool = False,
			head_mask: Optional[torch.Tensor] = None,
	) -> Union[tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor]]:
		mixed_query_layer = self.query(hidden_states)

		key_layer = self.transpose_for_scores(self.key(hidden_states))
		value_layer = self.transpose_for_scores(self.value(hidden_states))
		query_layer = self.transpose_for_scores(mixed_query_layer)

		pos_ids = self.get_position_ids(hidden_states, masks=position_mask)
		key_layer = self.apply_rotary_embeddings(key_layer, pos_ids)
		query_layer = self.apply_rotary_embeddings(query_layer, pos_ids)

		# Scaled Dot Product Attention often doesn't expose attention weights directly.
		attention_inference: Callable = eager_attention_forward
		if self.config._attn_implementation != "eager":
			if self.config._attn_implementation == "sdpa" and output_attentions:
				logger.warning_once(
					"`torch.nn.funcitonal.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to "
					'eager attention. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
				)
			else:
				attention_inference = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

			# Attention Calculation: calls selected attention function with prepared queries, keys, values, head mask and other paraments
			context_layer, attention_probs = attention_inference(
				self, 
				query_layer,
				key_layer, 
				value_layer, 
				head_mask, 
				is_casual=self.is_causal,
				scaling=self.scaling,
				dropout=0.0 if not self.training else self.dropout_prob
			)
			# Reshapes attention output: (batch_size, num_heads, sequence_length, head_dim) -> (batch_size, sequence_length, all_head_size)
			new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
			# Applies the final linear projection
			context_layer = self.proj(context_layer.reshape(new_context_layer_shape))

			outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

			return outputs

"""
^^^ Attention layer that takes patch embeddings, calculates their 3D positions, 
applies Rotary Positional Embeddings to queries adn keys to incorporate spatial 
and temporal relationships, performs multi-head scaled dot-product attention, 
and the projects the output. Ensures positional awareness crucial for video processing.
"""
