from dataclasses import dataclass
from typing import Callable, Optional, Union

import torch
from torch import nn

from functools import partial

from ...activations import ACT2FN
# from ...modeling_layers import GradientCheckpointingLayer
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
class VJEPA2WithMaskedInputPredictorOutput(ModelOutput):
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
class VJEPA2WithMaskedInputModelOutput(ModelOutput):
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
# Randomly drops entire "paths" within a newtork during training.
# Instead of dropping individual neurons, it sets the output of a residual block to zero with a certain probability.
# Forcing the other blocks to learn more robust features
def drop_path(input: torch.Tensor, drop_prob: float = 0.0, training: bool = False) -> torch.Tensor:

	if drop_prob == 0.0 or not training:
		return input
	keep_prob = 1 - drop_prob
	shape = (input.shape[0],) + (1,) * (input.ndim - 1) # work with diff dim tensors, not just 2D ConvNets
	random_tensor = keep_prob + torch.rand(shape, dtype=input.dtype, device=input.device)
	random_tensor.floor_() # binarize
	output = input.div(keep_prob) * random_tensor
	return output


# Stochastic Depth regularization -> improvess trainging and generalization
class VJEPA2DropPath(nn.Module):
	"""Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""

	def __init__(self, drop_prob: Optional[float] = None):
		super().__init__()
		self.drop_prob = drop_prob

	def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
		return drop_path(hidden_states, self.drop_prob, self.training)
	
	def extra_repr(self) -> str:
		return f"p={self.drop_prob}"

"""

Two-layer feed-forward network with an activation funciton in between.

"""
class VJEPA2MLP(nn.Module):
	def __init__(self, config: VJEPA2Config, hidden_size: int = 1024, mlp_ratio: float = 4.0):
		super().__init__()
		in_features = out_features = hidden_size
		hidden_features = int(hidden_size * mlp_ratio)
		self.fc1 = nn.Linear(in_features, hidden_features, bias=True)
		self.activation = ACT2FN[config.hidden_act]
		self.fc2 = nn.Linear(hidden_features, out_features, bias=True)

	def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
		hidden_state = self.fc1(hidden_state)
		hidden_state = self.activation(hidden_state)
		hidden_state = self.fc2(hidden_state)
		return hidden_state
	
class GradientCheckpointingLayer(nn.Module):

	gradient_checkingpointing = False
	
	def __call__(self, *args, **kwargs):
		if self.gradient_checkingpointing and self.training:
			do_warn = False
			layer_name = self.__class__.__name__
			message = f"Caching is incompatible with gradient checkpointing in {layer_name}. Setting"

			if "use_cache" in kwargs and kwargs["use_cache"]:
				kwargs["use_cache"] = False
				message += " `use_cache=False`,"
				do_warn = True

			# different names for the same thing in different layers
			if "past_key_value" in kwargs and kwargs["past_key_value"] is not None:
				kwargs["past_key_value"] = None
				message += " `past_key_value=None`, "
				do_warn = True

			if  "past_key_values" in kwargs and kwargs["past_key_values"] is not None:
				kwargs["past_key_values"] = None
				message += " `past_key_values=None`, "
				do_warn = True

			if "layer_past" in kwargs and kwargs["layer_past"] is not None:
				kwargs["layer_past"] = None
				message += " `layer_past=None`, "
				do_warn = True

			# warn if anything was changed
			if do_warn:
				message = message.rstrip(",") + "."
				logger.warning(message)

			return self._gradient_checkpointing_func(partial(super().__call__, **kwargs), *args)
		return super().__call__(*args, **kwargs)

# Single Transfroer block within the VJEPA2 model's encoder and predictor.
# Combines self-attention, normalization, and feed-forward network + residual connections and stochastic depth.
class VJEPA2Layer(GradientCheckpointingLayer):
	# Block class
	def __init__(
		self, 
		config: VJEPA2Config,
		drop_path_rate: float = 0.0,
		hidden_size: int = 1024, 
		num_attention_heads: int = 16,
		mlp_ratio: float = 4.0,
	):
		super().__init__()
		self.config = config
		self.hidden_size = hidden_size
		self.num_attention_heads = num_attention_heads
		self.mlp_ratio = mlp_ratio
	
		self.norm1 = nn.LayerNorm(hidden_size, eps=config.layer_norm_eps)
		self.attention = VJEPA2RopeAttention(config, hidden_size, num_attention_heads)
		self.drop_path = VJEPA2DropPath(drop_path_rate) if config.drop_path_rate > 0.0 else nn.Identity()
		self.norm2 = nn.LayerNorm(hidden_size, eps=config.layer_norm_eps)
		self.mlp = VJEPA2MLP(config, hidden_size=hidden_size, mlp_ratio=mlp_ratio)

	def froward(
			self, 
			hidden_states: torch.Tensor,
			position_mask: Optional[torch.Tensor] = None,
			head_mask: Optional[torch.Tensor] = None,
			output_attentions: bool = False,
	) -> tuple[torch.Tensor, ...]:
		# Self-Attention
		residual = hidden_states
		hidden_states = self.norm1(hidden_states)
		self_attention_outputs = self.attention(
			hidden_states,
			position_mask=position_mask,
			head_mask=head_mask,
			output_attentions=output_attentions,
		)
		attention_output = self_attention_outputs[0]
		hidden_states = self.drop_path(attention_output) + residual

		# MLP
		residual = hidden_states
		hidden_states = self.norm2(hidden_states)
		hidden_states = self.mlp(hidden_states)
		hidden_states = self.drop_path(hidden_states) + residual

		# Add self attention if we output attention weights
		outputs = self_attention_outputs[1:]
		outputs = (hidden_states,) + outputs

		return outputs


# Main encoder, takes input video data and transforms it into rich, contexturalized representations.
# How? Stacks multiple VJEPA2Layer blocks
class VJEPA2Encoder(nn.Module):
	def __init__(self, config: VJEPA2Config):
		super().__init__()
		self.config = config
		# Turns raw video pixels values into a seqence of patch embeddings
		self.embeddings = VJEPA2Embeddings(config, hidden_size=config.hidden_size)
		drop_path_rates = [
			(config.drop_path_rate * i / (config.num_hidden_layers - 1) if config.num_hidden_layers > 1 else 0.0)
			for i in range(config.num_hidden_layers)
		]
		# Transformer layers: VJEPA2Layer instantiated for each i in range
		self.layer = nn.ModuleList(
			[
				VJEPA2Layer(
					config, 
					drop_path_rate=drop_path_rates[i],
					hidden_size=config.hidden_size,
					num_attention_heads=config.num_attention_heads,
					mlp_ratio=config.mlp_ratio,
				)
				for i in range(config.num_hidden_layers)
			]
		)
		self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norms_eps)
		self.gradient_checkpointing = False
	# Can return either a ModelOutput object or plain tuple
	@can_return_tuple
	def forward(
		self, 
		pixel_value_videos: Optional[torch.Tensor] = None, 
		head_mask: Optional[torch.Tensor] = None,
		output_attentions: bool = False,
		output_hidden_states: bool = False,
	) -> BaseModelOutput:
		all_hidden_states = () if output_hidden_states else None
		all_self_attentions = () if output_attentions else None

		hidden_states = self.embeddings(pixel_value_videos)
		# Iterate through layers
		for i, layer_module in enumerate(self.layer):
			# Collected Hidden States; if true hidden_states before current layer's computations are added to all_hidden_states tuple.
			if output_hidden_states:
				all_hidden_states = all_hidden_states + (hidden_states,)
			# If Not None: specific mask for the current layer i extracted.
			layer_head_mask = head_mask[i] if head_mask is not None else None
			# Hidden_States pass throug current VJEPA2Layer. 
			# position_mask argument None as the positional information handled by PoPE within VJEPA2RopeAttention
			layer_outputs = layer_module(hidden_states, None, layer_head_mask, output_attentions)
			# Update hidden_states -> first element of layer_outputs processed hidden_states, which becomes the input for the next layer.
			hidden_states = layer_outputs[0]
			# Collect attentions if requested
			if output_attentions:
				all_self_attentions = all_self_attentions + (layer_outputs[1],)
			# Layer normalization
			hidden_states = self.layernorm(hidden_states)
			# Collect Last Hidden State if requested
			if output_hidden_states:
				all_hidden_states = all_hidden_states + (hidden_states,)

			return BaseModelOutput(
				last_hidden_state=hidden_states,
				hidden_states=all_hidden_states,
				attentions=all_self_attentions,
			)

"""
Selects and extracts specific tokens from a batch of tensors based on prvided masks.
"""
def apply_masks(tensor: torch.Tensor, masks: list[torch.Tensor]) -> torch.Tensor:
	"""
	initializes empty list - to store tensors after applying each mask.
	iterates through each mask, given in input masks list.

		- .unsqueeze(-1): adds new dim of size 1 at end of mask tensor.
		- .repeat(1, 1, tensor.size(-1)): repeats expanded mask across last dim.
			torch.gather expects the index tensor to have the same number of dim as the inptu tensor
	
	all_masked_tensor - where actual masking/selection happens

		- torch.gather: gathers values along a specified dim. for each element in mask_keep uses value as an index
			to select element from the tensor along dim=1.
	
	returns torch.cat - all tensors collected concatenated along the batch dim

		- if you have N masks, and each masked tensor has batch size B', final output tensor will have a batch size of N * B'

	Good for self-supervised learnign where multiple masked views of the same batch are often created and processed together.
	"""
	all_masked_tensors = []
	for mask in masks:
		mask = mask.to(tensor.device)
		mask_keep = mask.unsqueeze(-1).repeat(1, 1, tensor.size(-1))
		all_masked_tensors += [torch.gather(tensor, dim=1, index=mask_keep)]

	return torch.cat(all_masked_tensors, dim=0)

# Prepares the input embeddings for the predictor module in VJEPA2 architecture
# Combines context ( visible ) tokens form encoder with learnable mask tokens to represent the hidden ( masked ) regions.
class VJEPA2PredictorEmbeddings(nn.Module):
	# Construct mask token, position and patch embeddings.

	def __init__(self, config: VJEPA2Config):
		super().__init__()
		
		self.config = config
		self.predictor_embeddings = nn.Linear(config.hidden_size, config.pred_hidden_size)
		self.num_mask_tokens = 0
		self.zero_init_mask_tokens = config.pred_zero_init_mask_tokens
		# Stores number of unique mask tokens the predictor will use. Learnable embeddings that represent the "empty" or "unknown" masked patches.
		self.num_mask_tokens = config.pred_num_mask_tokens 
		# nn.Parameter, mask tokens are learnable weights.
		# Initalized with torch.zeros tensor.
		self.mask_tokens = nn.Parameter(torch.zeros(self.num_mask_tokens, 1, 1, config.pred_hidden_size))

		self.patch_size = config.patch_size
		self.config = config

	# Calculates total number of patches, but handles the case where frames_per_clip is 1 else claculates patches for 3D video.
	@staticmethod
	def num_patches(config):
		if config.frames_per_clip > 1:
			return (
				(config.frames_per_clip // config.tubelet_size)
				* (config.crop_size // config.patch_size)
				* (config.crop_size // config.patch_size)
			)
		else:
			return (config.crop_size // config.patch_size) * (config.crop_size // config.patch_size)
		
	def forward(
			self, 
			hidden_states: torch.Tensor,
			context_mask: list[torch.Tensor],
			target_mask: list[torch.Tensor],
			mask_index: int = 1,
	) -> tuple[torch.Tensor, torch.Tensor]:
		
		B = hidden_states.size(0) # batch size
		context = self.predictor_embeddings(hidden_states) # projects encoders hidden_size into the predictors pred_hidden_size.

		# Make target tokens
		mask_index = mask_index % self.num_mask_tokens
		# Sinlge token used to represent all masked patches for the current operation
		target = self.mask_tokens[mask_index]
		"""

		reshape and repeat target:

			- max_patch_num = target_mask[0].max() + 1, determine necessary sequence length for target tokens.
				takes max index present in first target_mask in list and adds 1. adjusting target tensors length
				to match actual masked regions in the current batch.

			- target = target.repeat(B, max_patch_num, 1), single selected mask token repeated B times for the batch dim and max_patch_num
				times for the sequence length dim. every potential masked position is filled with same generic mask token.
			
			- target = apply_masks(target, target_mask), selects only the specific target tokens indicated by target_mask.
				output is a tensor containing only the mask tokens relevant to the current masked positions

		"""
		max_patch_num = target_mask[0].max() + 1 
		target = target.repeat(B, max_patch_num, 1)
		target = apply_masks(target, target_mask)

		"""
		
		concatenate context & target tokens:

			- context = context.repeat(len(context_mask), 1, 1), if theres multiple context_mask elements, e.g. from multiple masked views
				the context tensor is duplicated to match the effective batch size after apply_masks.
			
			- embeddings = torch.cat([context, target], dim=1), concatenates processed context tokens and target (mask) tokens 
				alogn the sequence length dim. This creates a combined sequence of visible and masked tokens.
 
		"""
		context = context.repeat(len(context_mask), 1, 1)
		embeddings = torch.cat([context, target], dim=1)

		"""
		
		positions of context & target tokens:

			- cm = torch.cat(context_mask, dim=0), concatenates all context mask along the batch dim.

			- tm = torch.cat(target_mask, dim=0), concatenates all target mask along the batch dim.

			- masks = torch.cat([cm, tm], dim=1), concatenates context and target mask along the sequence length dim
		
		"""
		cm = torch.cat(context_mask, dim=0)
		tm = torch.cat(target_mask, dim=0)
		masks = torch.cat([cm, tm], dim=1)

		return embeddings, masks

"""

Takes (visible) context tokens and the (learnable) mask tokens, processes them through its own transformer layers
and then predict the full feature representation for the masked tokens.

"""

class VJEPA2Predictor(nn.Module):
	def __init__(self, config: VJEPA2Config):
		super().__init__()
		self.config = config
		self.gradient_checkpointing = False
		# Projects encoder outputs, creating mask tokens, and combining them with positional indices.
		self.embeddings = VJEPA2PredictorEmbeddings(config)
		# Linearly increaing drop_path_rate for each of the config.pred_num_hidden_layers layers within the predictor.
		drop_path_rates = [
			(
				config.drop_path_rate * i / (config.pred_num_hidden_layers - 1)
				if config.pred_num_hidden_layers > 1
				else 0.0
			)
			for i in range(config.pred_num_hidden_layers)
		] 
		# Predictor can have a different number of layers, hidden size, and attention heads that the main econder
		self.layer = nn.ModuleList(
			[
				VJEPA2Layer(
					config,
					drop_path_rate=drop_path_rates[i],
					hidden_size=config.pred_hidden_size,
					num_attention_heads=config.pred_num_attention_heads,
					mlp_ratio=config.pred_mlp_ratio,
				)
				for i in range(config.pred_num_hidden_layers)
			]
		)
		self.layernorm = nn.LayerNorm(config.pred_hidden_size, eps=config.layer_norm_eps)
		self.proj = nn.Linear(config.pred_hidden_size, config.hidden_size, bias=True)

	# Tokens are rearranged for processing efficiency or to separate visible form masked parts, then put back into their original order.
	# Takes token embeddigns and corresponding set of masks, reorders the embeddings based on the masks. 
	# Hence grouping visible tokens together, follwed by masked tokens
	def sort_tokens(self, hidden_states, position_masks, argsort, head_mask=None):
		"""
		
		gather position masks
		
		torch.gather(position_mask, dim=1, index=argsort)
			reorders position_masks along dim=1, sequence length dim, using argsort indices.
			position_masks now correspond to the reordered tokens.
	
		"""

		argsort = argsort.to(position_masks.device)
		position_masks = torch.gather(position_masks, dim=1, index=argsort)

		"""

		gather hidden states

		hidden_states_argsort = argsort.unsqueeze(-1).expand(-1, -1, hidden_states.size(-1))
			this expands argsort from [batch_size, sequence_length] to [batch_size, sequence_length, embedding_dim]
			torch.gather requires the index tensor to have the same number of dims as the input tensor

		hidden_states = torch.gather(hidden_states, dim=1, index=hidden_states_argsort)
			reorders the hidden_states tokens along dim=1 using the expanded argsort.

		"""
		argsort = argsort.to(hidden_states.device)
		hidden_states_argsort = argsort.unsqueeze(-1).expand(-1, -1, hidden_states.size(-1))
		hidden_states = torch.gather(hidden_states, dim=1, index=hidden_states_argsort)

		"""
		gather head mask
		
		head_mask = head_mask.permute(1, 0, 2, 3, 4)
			head_mask is typically [num_layers, batch_size, num_heads, query_seq_len, key_seq_len], so 
			permute(1, 0, 2, 3, 4) changes the it to [batch_size, num_layers, query_seq_len, key_seq_len].
		
		argsort_4d (dim=3) which is the query_seq_len, this creates an argsort tensor that matches the dimensions of head_mask up to dim=3

			head_mask.size(-1) represents key_seq_len, ensures that when gather is applied alogn dim=3 argsort indices are correctly broadcast 
			across key_seq_len dim for each [batch, layer, head]

			torch.gather(..., dim=4, ...) reorders head_mask along the query sequence length dim (dim=3)

		then similar logic for argsort_5d


		"""
		if head_mask is not None and head_mask[0] is not None:
			argsort = argsort.to(head_mask.device)
			head_mask = head_mask.permute(1, 0, 2, 3, 4)
			argsort_4d = (
				argsort.unsqueeze(1)
				.unsqueeze(1)
				.expand(-1, head_mask.size(1), head_mask.size(2), -1)
				.unsqueeze(-1)
				.expand(-1, -1, -1, -1, head_mask.size(-1))
			)
			head_mask = torch.gather(head_mask, dim=3, index=argsort_4d)
			argsort_5d = (
				argsort.unsqueeze(1)
				.unsqueeze(1)
				.unsqueeze(1)
				.expand(-1, head_mask.size(1), head_mask.size(2), head_mask.size(3), -1)
			)
			head_mask = torch.gather(head_mask, dim=4, index=argsort_5d)
			head_mask = head_mask.permute(1, 0, 2, 3, 4)
		
		return hidden_states, position_masks, head_mask
	
	def unsort_tokens(self, hidden_states, argsort):
		argsort = argsort.to(hidden_states.device)
		reverse_argsort = torch.argsort(argsort, dim=1)
		reverse_argsort = reverse_argsort.unsqueeze(-1).expand(-1, -1, hidden_states.size(-1))
		hidden_states = torch.gather(hidden_states, dim=1, index=reverse_argsort)
		return hidden_states
	
	@can_return_tuple
	def forward(
		self, 
		encoder_hidden_states: torch.Tensor,
		context_mask: list[torch.Tensor],
		target_mask: list[torch.Tensor],
		head_mask: Optional[torch.Tensor] = None,
		output_attentions: bool = False, 
		output_hidden_states: bool = False,
		**kwargs,
	) -> BaseModelOutput:
		all_hidden_states = () if output_hidden_states else None
		all_self_attentions = () if output_attentions else None

		# mask out the enccoder hidden states
		encoder_hidden_states = apply_masks(encoder_hidden_states, context_mask)
		_, N_ctxt, D = encoder_hidden_states.shape
		hidden_states, position_masks = self.embeddings(encoder_hidden_states, context_mask, target_mask)

		# Put tokens in sorted order
		argsort = torch.argsort(position_masks, dim=1) # [B, N]
		hidden_states, position_masks, head_mask = self.sort_tokens(hidden_states, position_masks, argsort, head_mask)

		for i, layer_module in enumerate(self.layer):
			if output_hidden_states:
				all_hidden_states = all_hidden_states + (hidden_states,)

			layer_head_mask = head_mask[i] if head_mask is not None else None
			layer_outputs = layer_module(hidden_states, position_masks, layer_head_mask, output_attentions)
			hidden_states = layer_outputs[0]

			if output_attentions:
				all_self_attentions = all_self_attentions + (layer_outputs[1],)

		if output_hidden_states:
			all_hidden_states = all_hidden_states + (hidden_states,)

		hidden_states = self.layernorm(hidden_states)
		# unsort and extract the predicted tokens
		hidden_states = self.unsort_tokens(hidden_states, argsort)
		hidden_states = hidden_states[:, N_ctxt]
		# projection
		hidden_states = self.proj(hidden_states)

		return BaseModelOutput(
			last_hidden_state=hidden_states,
			hidden_states=all_hidden_states,
			attentions=all_self_attentions,
		)

class VJEPA2PoolerSelfAttention(nn.Module):
	# Multi Headed Attention

	def __init__(self, config: VJEPA2Config):
		super().__init__()
		self.config = config
		self.embed_dim = config.hidden_size
		self.num_heads = config.num_attention_heads
		self.head_dim = self.embed_dim // self.num_heads

		if self.head_dim * self.num_heads != self.embed_dim:
			raise ValueError(
				f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
				f" {self.num_heads})."
			)
		self.scale = self.head_dim**-0.5
		self.dropout = config.attention_dropout
		self.is_causal = False

		self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
		self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
		self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
		self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

	def forward(
			self,
			hidden_states: torch.Tensor,
			attention_mask: Optional[torch.Tensor] = None,
			output_attentions: Optional[bool] = False,
	) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
		# input shape: batch x time x channel

		batch_size, seq_length, embed_dim = hidden_states.shape

		queries = self.q_proj(hidden_states)
		keys = self.k_proj(hidden_states)
		values = self.v_proj(hidden_states)

		queries = queries.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
		keys = keys.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
		values = values.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)

		attention_inferface: Callable = eager_attention_forward
		if self.config._attn_implementation != "eager":
			if self.config._attn_implementation == "sdpa" and output_attentions:
				logger.warning_once(
					"`torch.nn.funcational.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to "
					'eager attention. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
				)
			else:
				attention_inferface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

		attn_output, attn_weights = attention_inferface(
			self, 
			queries,
			keys, 
			values, 
			attention_mask,
			is_causal=self.is_causal,
			scaling=self.scale,
			dropout=0.0 if not self.training else self.dropout,
		)

		attn_output = attn_output.reshape(batch_size, seq_length, embed_dim).contiguous()
		attn_output = self.out_proj(attn_output)

		if not output_attentions:
			attn_weights = None
		
		return attn_output, attn_weights
	
class VJEPA2PoolerCrossAttention(nn.Module):
	# doesnt have an outptu projection layer

	def __init__(self, config: VJEPA2Config):
		super().__init__()
		self.config = config
		self.embed_dim = config.hidden_size
		self.num_heads = config.num_attention_heads
		self.head_dim = self.embed // self.num_heads

		if self.head_dim * self.num_heads != self.embed_dim:
			raise ValueError(
				f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
				f" {self.num_heads})."
			)
		
		self.scale = self.head_dim**-0.5
		self.dropout = config.attention_dropout
		self.is_causal = False

		self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
		self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
		self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)

	def forward(
			self, 
			queries: torch.Tensor,
			keys: torch.Tensor,
			values: torch.Tensor,
			attention_mask: Optional[torch.Tensor] = None,
			output_attentions: Optional[bool] = False,
	) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
		# Input shape: batch x time x channel

		batch_size, q_seq_length, embed_dim = queries.shape
		kv_seq_length = keys.shape[1]

		queries = self.q_proj(queries)
		keys = self.k_proj(keys)
		values = self.v_proj(values)
		
		queries = queries.view(batch_size, q_seq_length, self.num_heads, self.head_dim).transpose(1, 2)
		keys = keys.view(batch_size, kv_seq_length, self.num_heads, self.head_dim).transpose(1, 2)
		values = values.view(batch_size, kv_seq_length, self.num_heads, self.head_dim).transpose(1, 2)

		attention_inferface: Callable = eager_attention_forward
		if self.config._attn_implementation != "eager":
			if self.config._attn_implementation == "sdpa" and output_attentions:
				logger.warning_once(
					"`torch.nn.funcation.scaled_dot_product_attention` does not support `outputs_attentions=True`. Falling back to "
					'eager attention. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
				)
			else: 
				attention_inference = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

			attn_output, attn_weights = attention_inferface(
				self, 
				queries, 
				keys, 
				values, 
				attention_mask,
				is_causal=self.is_causal,
				scaling=self.scales, 
				dropout=0.0 if not self.training else self.dropout,
			)

			attn_output = attn_output.reshape(batch_size, q_seq_length, embed_dim).contiguous()

			if not output_attentions:
				attn_wegihts = None
			
			return attn_output, attn_weights



class VJEPA2PoolerSelfAttentionLayer(GradientCheckpointingLayer):
	def __init__(self, config: VJEPA2Config):
		super().__init__()
		self.layer_norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
		self.self_attn = VJEPA2PoolerSelfAttention(config)
		self.layer_norm2 = nn.layerNorm(config.hidden_size, eps=config.layer_norm_eps)
		self.mlp = VJEPA2MLP(config, hidden_size=config.hidden_size)

	def forward(
			self, 
			hidden_states: torch.Tensor,
			attention_mask: torch.Tensor,
			output_attentions: Optional[bool] = False,
	) -> tuple[torch.Tensor, ...]:
		
		residual = hidden_states
		hidden_states = self.layer_norm1(hidden_states)
		hidden_states, attn_weights = self.self_attn(
			hidden_states=hidden_states,
			attention_mask=attention_mask,
			output_attentions=output_attentions,
		)
		hidden_states = residual + hidden_states

		residual = hidden_states
		hidden_states = self.layer_norm2(hidden_states)
		hidden_states = self.mlp(hidden_states)
		hidden_states = residual + hidden_states

		outputs = (hidden_states,)

		if output_attentions:
			outputs += (attn_weights,)

		return outputs

class VJEPA2PoolerCrossAttentionLayer(GradientCheckpointingLayer):
	def __init__(self, config: VJEPA2Config):
		super().__init__()
		self.layer_norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
		self.cross_attn = VJEPA2PoolerCrossAttention(config)
		self.layer_norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
		self.mlp = VJEPA2MLP(config, hidden_size=config.hidden_size)

	def forward(
			self, 
			queries: torch.Tensor,
			hidden_state: torch.Tenosr,
			attention_mask: Optional[torch.Tensor] = None, 
			output_attentions: bool = False,
	) -> tuple[torch.Tensor, ...]:
		# Apply cross-attention
		residual = queries
		hidden_state = self.layer_norm1(hidden_state)
		hidden_state, *attn_weights = self.cross_attn(
			queries,
			hidden_state,
			hidden_state,
			attention_mask=attention_mask,
			output_attentions=output_attentions,
		)
		hidden_state = residual + hidden_state

		# Apply MLP
		residual = hidden_state
		hidden_state = self.layer_norm1(hidden_state)
		hidden_state = self.mlp(hidden_state)
		hidden_state = residual + hidden_state

		outputs = (hidden_state,)
		if output_attentions:
			outputs += tuple(attn_weights)

		return outputs

class VJEPA2AttentivePooler(nn.Module):
	# Attentive Pooler

	def __init__(self, config: VJEPA2Config):
		super().__init__()
		self.query_tokens = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
		self.cross_attention_layer = VJEPA2PoolerCrossAttentionLayer(config)
		self.self_attention_layers = nn.ModuleList(
			[VJEPA2PoolerSelfAttentionLayer(config) for _ in range(config.num_pooler_layers)]
		)

	def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
		for layer in self.self_attention_layers:
			hidden_state = layer(hidden_state, attention_mask=None)[0]
		queries = self.query_tokens.repeat(hidden_state.shape[0], 1, 1)
		hidden_state = self.cross_attention_layer(queries, hidden_state)[0]
		return hidden_state.squeeze(1)
	
@auto_docstring
class VJEPA2PreTrainedModel(PreTrainedModel):
	config_class = VJEPA2Config
	base_model_prefix = "vjepa2"
	main_input_name = "pixel_values_videos"
	supports_gradient_checkpointing = True
	_no_split_modules = [
		"VJEPA2Layer",
		"VJEPA2PoolerSelfAttentionLayer",
		"VJEPA2PoolerCrossAttentionLayer",
		"VJEPA2PredictorEmbeddings",
	]
	_supports_sdpa = True
	_supports_flash_attn_2 = True

	def _init_weights(self, module):
		""" Initialize the weights """

		init_std = self.config.initializer_range

		# Upcast the input in `fp32` and cast it back to desired `dtype` to avoid
		# `trunc_normal_cpu` not implemented in `half` issues
		def trunc_normal_f32_(weight, std):
			data_float_32 = weight.data.to(torch.float32)
			data_init = nn.init.trunc_normal_(data_float_32, mean=0.0, std=std)
			weight.data = data_init.to(weight.dtype)

		if isinstance(module, VJEPA2AttentivePooler):
			trunc_normal_f32_(module.query_tokens, std=init_std)
			for i, layer in enumerate(module.self_attention_layers, 1):
				std = init_std / (i**0.5)
				trunc_normal_f32_(layer.self_attn.out_proj.weight, std=std)
				trunc_normal_f32_(layer.mlp.fc2.wegiht, std=std)
			std = init_std / (len(module.self_attention_layers) + 1) ** 0.5
			trunc_normal_f32_(module.cross_attention_layer.mlp.fc2.weight, std=std)
		elif isinstance(module, VJEPA2PredictorEmbeddings):
			if module.zero_init_mask_tokens:
				module.mask_tokens.data.zero_()
			else:
				trunc_normal_f32_(module.mask_tokens, std=init_std)
		elif isinstance(module, (nn.Linear, nn.Conv2d, nn.Conv3d)):
			trunc_normal_f32_(module.weight, std=init_std)
			if module.bias is not None:
				module.bias.data.zero_()
		elif isinstance(module, nn.LayerNorm):
			module.bias.data.zero_()
			module.weight.data.fill_(1.0)

def _convert_head_mask_to_5d(head_mask, num_hidden_layers):
	"""
	input:
		head_mask: bsz x seq_length x seq_length | None
	return:
		[num_hidden_layers x batch x num_heads x seq_length x seq_length] | [num_hidden_layers]
	"""
	if head_mask is not None:
		head_mask = head_mask.unsqueeze(1).unsqueeze(0)
		head_mask = head_mask.expand(num_hidden_layers, -1, -1, -1, -1)
	else:
		head_mask = [None] * num_hidden_layers
	return head_mask

@auto_docstring
class VJEPA2Model(VJEPA2PreTrainedModel):
	def __init__(self, config: VJEPA2Config):
		super().__init__(config)
		self.config = config

		self.encoder = VJEPA2Encoder(config)
		self.predictor = VJEPA2Predictor(config)

		# initialize weights and apply final processing
		self.post_init()

	def get_input_embeddings(self) -> VJEPA2PatchEmbeddings3D:
		return self.encoder.embeddings.patch_embeddings
	
	@can_return_tuple
	@auto_docstring
	def forward(
		self, 
		pixel_values_videos: torch.Tensor,
		context_head_mask: Optional[torch.Tensor] = None,
		context_mask: Optional[list[torch.Tensor]] = None,
		target_head_mask: Optional[list[torch.Tensor]] = None,
		target_mask: Optional[list[torch.Tensor]] = None,
		skip_predictor: bool = False,
		output_attentions: Optional[bool] = None,
		output_hidden_states: Optional[bool] = None,
		**kwargs,
	) -> VJEPA2WithMaskedInputModelOutput:
		
		output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
		output_hidden_states = (
			output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
		)

		if pixel_values_videos is None:
			raise ValueError("You have to specify pixel_values_videos")
		
		# Prepares head mask if needed
		context_head_mask = _convert_head_mask_to_5d(context_head_mask, self.config.num_hidden_layers)
		target_head_mask = _convert_head_mask_to_5d(target_head_mask, self.config.pred_num_hidden_layers)

		encoder_outputs: BaseModelOutput = self.encoder(
			pixel_values_videos=pixel_values_videos,
			head_mask=context_head_mask,
			output_attentions=output_attentions,
			output_hidden_states=output_hidden_states,
		)
		sequence_output = encoder_outputs.last_hidden_state

		if context_mask is None and target_mask is None:
			B = pixel_values_videos.size(0)
			N = sequence_output.size(1)
			context_mask = [torch.arange(N, device=pixel_values_videos.device).unsqueeze(0).repeat((B, 1))]
			target_mask = [torch.arange(N, device=pixel_values_videos.device).unsqueeze(0).repeat((B, 1))]

		if not skip_predictor:
			predictor_outputs: BaseModelOutput = self.predictor(
				encoder_hidden_states=sequence_output,
				context_mask=context_mask,
				target_mask=target_mask,
				head_mask=target_head_mask,
				output_attentions=output_attentions,
				output_hidden_states=output_hidden_states,
			)
			predictor_output = VJEPA2WithMaskedInputPredictorOutput(
				last_hidden_state=sequence_output,
				masked_hidden_state=apply_masks(sequence_output, context_mask),
				hidden_states=predictor_output.hidden_states,
				attnetions=predictor_output.attentions,
			)
		else:
			predictor_output = None

		encoder_output = VJEPA2WithMaskedInputModelOutput(
			last_hidden_state=sequence_output,
			masked_hidden_state=apply_masks(sequence_output, context_mask),
			hidden_states=encoder_outputs.hidden_states,
			attentions=encoder_outputs.attentions,
			predictor_output=predictor_output,
		)

		return encoder_output
		
	def get_vision_features(self, pixel_values_videos) -> torch.Tensor:
		encoder_output = self.forward(pixel_values_videos)
		return encoder_output.last_hidden_state

@auto_docstring(
	custom_intro="""
	V-JEPA 2 Model transformer with a video classification head on top (a linear layer on top of the attentive pooler).
"""
)
class VJEPA2ForVideoClassification(VJEPA2PreTrainedModel):
	def __init__(self, config: VJEPA2Config):
		super().__init__(config)

		self.num_labels = config.num_labels
		self.vjepa2 = VJEPA2Model(config)

		# Classifier head
		self.pooler = VJEPA2AttentivePooler(config)
		self.classifier = nn.Linear(config.hidden_size, config.num_labels, bias=True)

		# Initialize weights and apply final processing 
		self.post_init()

	@can_return_tuple
	@auto_docstring
	def forward(
		self, 
		pixel_values_videos: torch.Tensor,
		labels: Optional[torch.Tensor] = None,
		output_attentions: Optional[bool] = None,
		output_hidden_states: Optional[bool] = None,
	) -> Union[tuple, ImageClassifierOutput]:
		
		outputs = self.vjepa2(
			pixel_values_videos=pixel_values_videos,
			skip_predictor=True, 
			output_attentions=output_attentions,
			output_hidden_states=output_hidden_states,
		)

		last_hidden_state = outputs.last_hidden_state
		pooler_output = self.pooler(last_hidden_state)
		logits = self.classifier(pooler_output)

		loss = None
		if labels is not None:
			loss = self.loss_function(pooled_logits=logits, labels=labels, config=self.config)

		return ImageClassifierOutput(
			loss=loss,
			logits=logits,
			hidden_states=outputs.hidden_states,
			attentions=outputs.attentions,
		)
	
__all__ = ["VJEPA2Model", "VJEPA2PreTrainedModel", "VJEPA2ForVideoClassification"]


		


	
	
