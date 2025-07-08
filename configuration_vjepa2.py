"""VJEPA 2 model configuration"""

from ...configuration_utils import PretrainedConfig


class VJEPA2Config(PretrainedConfig):
    
    model_type = "vjepa2"

    def __init__(
        self,
        patch_size=16,
        crop_size=256,
        frames_per_clip=64,
        tubelet_size=2,
        hidden_size=1024,
        in_chans=3,
        num_attention_heads=16,
        num_hidden_layers=24,
        drop_path_rate=0.0,
        mlp_ratio=4.0,
        layer_norm_eps=1e-6,
        qkv_bias=True,
        attention_probs_dropout_prob=0.0,
        hidden_act="gelu",
        initializer_range=0.02,
        attention_dropout=0.0,
        num_pooler_layers=3,
        # predictor params
        pred_hidden_size=384,
        pred_num_attention_heads=12,
        pred_num_hidden_layers=12,
        pred_num_mask_tokens=10,
        pred_zero_init_mask_tokens=True,
        pred_mlp_ratio=4.0,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.crop_size = crop_size
        self.frames_per_clip = frames_per_clip
        self.patch_size = patch_size
        self.tubelet_size = tubelet_size
        self.hidden_size = hidden_size
        self.in_chans = in_chans
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.drop_path_rate = drop_path_rate
        self.mlp_ratio = mlp_ratio
        self.layer_norm_eps = layer_norm_eps
        self.qkv_bias = qkv_bias
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.image_size = crop_size
        self.attention_dropout = attention_dropout
        self.num_pooler_layers = num_pooler_layers
        # predictor params
        self.pred_hidden_size = pred_hidden_size
        self.pred_num_attention_heads = pred_num_attention_heads
        self.pred_num_hidden_layers = pred_num_hidden_layers
        self.pred_num_mask_tokens = pred_num_mask_tokens
        self.pred_zero_init_mask_tokens = pred_zero_init_mask_tokens
        self.pred_mlp_ratio = pred_mlp_ratio


__all__ = ["VJEPA2Config"]