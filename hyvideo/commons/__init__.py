# Licensed under the TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5/blob/main/LICENSE
#
# Unless and only to the extent required by applicable law, the Tencent Hunyuan works and any
# output and results therefrom are provided "AS IS" without any express or implied warranties of
# any kind including any warranties of title, merchantability, noninfringement, course of dealing,
# usage of trade, or fitness for a particular purpose. You are solely responsible for determining the
# appropriateness of using, reproducing, modifying, performing, displaying or distributing any of
# the Tencent Hunyuan works or outputs and assume any and all risks associated with your or a
# third party's use or distribution of any of the Tencent Hunyuan works or outputs and your exercise
# of rights and permissions under this agreement.
# See the License for the specific language governing permissions and limitations under the License.

import os
import torch
from itertools import repeat
from contextlib import contextmanager
from torch import nn
import collections.abc

def _ntuple(n):
    """Create a function that converts input to n-tuple."""
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            x = tuple(x)
            if len(x) == 1:
                x = tuple(repeat(x[0], n))
            return x
        return tuple(repeat(x, n))
    return parse

# Convenience functions for common tuple sizes
to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)
to_3tuple = _ntuple(3)
to_4tuple = _ntuple(4)

PRECISION_TO_TYPE = {
    'fp32': torch.float32,
    'fp16': torch.float16,
    'bf16': torch.bfloat16,
}

# Default generation pipeline configurations
PIPELINE_CONFIGS = {
    'omniweaving': {
        'guidance_scale': 6.0,
        'embedded_guidance_scale': None,
        'flow_shift': 7.0,
    },
    'omniweaving2': {
        'guidance_scale': 6.0,
        'embedded_guidance_scale': None,
        'flow_shift': 5.0,
    }
}


def is_flash2_available():
    try:
        from flash_attn import flash_attn_varlen_qkvpacked_func
        return True
    except Exception:
        return False

def is_flash3_available():
    try:
        from flash_attn_interface import flash_attn_varlen_func as flash_attn_varlen_func_v3  # noqa: F401
        return True
    except Exception:
        return False

def is_flash_available():
    return is_flash2_available() or is_flash3_available()

def is_sparse_attn_supported():
    return 'nvidia h' in torch.cuda.get_device_properties(0).name.lower()

def is_sparse_attn_available():
    if not is_sparse_attn_supported():
        return False
    try:
        from flex_block_attn import flex_block_attn_func  # noqa: F401
        return True
    except Exception:
        return False

def is_angelslim_available():
    try:
        import angelslim
        return True
    except Exception:
        return False

def maybe_fallback_attn_mode(attn_mode):
    """
    Determine the final attention mode based on configuration and availability.
    
    Args:
        attn_mode: Requested attention mode
        infer_state: Inference configuration object (optional)
        block_idx: Current block index (optional)
    
    Returns:
        Final attention mode to use
    """
    import warnings
    original_attn_mode = attn_mode

    if attn_mode in ('flex-block-attn'):
        from hyvideo.commons import is_sparse_attn_available
        if not is_sparse_attn_available():
            raise ValueError(f"{attn_mode} is not available for your GPU or flex-block-attn is not properly installed.")
    
    enable_sageattn = attn_mode == 'sageattn'
    
    assert not (enable_sageattn and attn_mode == 'flex-block-attn'), \
        ("SageAttention cannot be used with flex-block-attn mode. "
         "Please disable enable_sageattn or use a different attention mode.")
    
    # Use SageAttention if configured
    if attn_mode == 'sageattn':
        try:
            from sageattention import sageattn
        except Exception:
            attn_mode = 'flash'
    # Handle flash attention modes
    if attn_mode == 'flash':
        if is_flash3_available():
            attn_mode = 'flash3'
        elif is_flash2_available():
            attn_mode = 'flash2'
        else:
            attn_mode = 'torch'
    elif attn_mode == 'flash3':
        if not is_flash3_available():
            attn_mode = 'torch'
    elif attn_mode == 'flash2':
        if not is_flash2_available():
            attn_mode = 'torch'
    if attn_mode != original_attn_mode and not ('flash' in original_attn_mode and 'flash' in attn_mode):
        warnings.warn(f"Falling back from `{original_attn_mode}` to `{attn_mode}` because `{original_attn_mode}` is not properly installed.")
    return attn_mode

@contextmanager
def auto_offload_model(models, device, enabled=True):
    # from diffusers.hooks.group_offloading import _is_group_offload_enabled
    if enabled:
        if isinstance(models, nn.Module):
            models = [models]
        for model in models:
            if model is not None:
                model.to(device)
    yield
    if enabled:
        for model in models:
            if model is not None:
                model.to(torch.device('cpu'))

def get_gpu_memory(device=None):
    if not torch.cuda.is_available():
        return 0
    device = device if device is not None else torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device)
    if hasattr(torch.cuda, 'get_per_process_memory_fraction'):
        memory_fraction = torch.cuda.get_per_process_memory_fraction()
    else:
        memory_fraction = 1.0
    return props.total_memory * memory_fraction

def get_rank():
    return int(os.environ.get('RANK', '0'))
