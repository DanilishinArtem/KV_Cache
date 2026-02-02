from specache.utils import Quen2Attention_init, SpeCache_forward, CausalLM_forward
from transformers.models.qwen2 import modeling_qwen2


def patch_attention(specache_config):
    def init_wrapper(self, config, layer_idx):
        Quen2Attention_init(self, config, layer_idx, specache_config)

    modeling_qwen2.Qwen2Attention.__init__ = init_wrapper
    modeling_qwen2.Qwen2Attention.forward = SpeCache_forward
    modeling_qwen2.Qwen2ForCausalLM.forward = CausalLM_forward