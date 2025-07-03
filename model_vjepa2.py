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
		return tuples(output)


