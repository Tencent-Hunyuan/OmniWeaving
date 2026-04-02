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

if 'PYTORCH_CUDA_ALLOC_CONF' not in os.environ:
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import datetime
import json

import numpy as np
import loguru
import torch
import argparse
import einops
import imageio
from torchvision import transforms
from torch import distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import get_model_state_dict

from hyvideo.pipelines.hunyuan_video_pipeline import HunyuanVideo_1_5_Pipeline
from hyvideo.commons import PIPELINE_CONFIGS, auto_offload_model
from hyvideo.utils.data_utils import generate_crop_size_list
from hyvideo.commons.parallel_states import initialize_parallel_state
from hyvideo.commons.infer_state import initialize_infer_state

parallel_dims = initialize_parallel_state(sp=int(os.environ.get('WORLD_SIZE', '1')))
torch.cuda.set_device(int(os.environ.get('LOCAL_RANK', '0')))

def save_video(video, path, fps=24):
    if video.ndim == 5:
        assert video.shape[0] == 1
        video = video[0]
    vid = (video * 255).clamp(0, 255).to(torch.uint8)
    vid = einops.rearrange(vid, 'c f h w -> f h w c')
    imageio.mimwrite(path, vid, fps=fps)

def rank0_log(message, level):
    if int(os.environ.get('RANK', '0')) == 0:
        loguru.logger.log(level, message)

def save_config(args, output_path, task):
    arguments = {}
    for key, value in vars(args).items():
        if not key.startswith('_') and not callable(value):
            try:
                json.dumps(value)
                arguments[key] = value
            except (TypeError, ValueError):
                arguments[key] = str(value)
    
    config = {
        'timestamp': datetime.datetime.now().isoformat(),
        'task': task,
        'output_path': output_path,
        'arguments': arguments
    }
    
    base_path, _ = os.path.splitext(output_path)
    config_path = f"{base_path}_config.json"
    
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    print(f"Saved generation config to: {config_path}")
    return config_path

def str_to_bool(value):
    """Convert string to boolean, supporting true/false, 1/0, yes/no.
    If value is None (when flag is provided without value), returns True."""
    if value is None:
        return True  # When --flag is provided without value, enable it
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        value = value.lower().strip()
        if value in ('true', '1', 'yes', 'on'):
            return True
        elif value in ('false', '0', 'no', 'off'):
            return False
    raise argparse.ArgumentTypeError(f"Boolean value expected, got: {value}")

def load_checkpoint_to_transformer(pipe, checkpoint_path):
    if not os.path.exists(checkpoint_path):
        raise ValueError(f"Checkpoint path does not exist: {checkpoint_path}")
    rank0_log(f"Loading checkpoint from {checkpoint_path}", "INFO")
    
    try:
        model_state_dict = get_model_state_dict(pipe.transformer)
        dcp.load(
            state_dict={"model": model_state_dict},
            checkpoint_id=checkpoint_path,
        )
        rank0_log("Transformer model state loaded successfully", "INFO")
    except Exception as e:
        rank0_log(f"Error loading checkpoint: {e}", "ERROR")
        raise

def load_lora_adapter(pipe, lora_path):
    rank0_log(f"Loading LoRA adapter from {lora_path}", "INFO")
    try:
        pipe.transformer.load_lora_adapter(
            pretrained_model_name_or_path_or_dict=lora_path,
            prefix=None,
            adapter_name="default",
            use_safetensors=True,
            hotswap=False,
        )
        rank0_log("LoRA adapter loaded successfully", "INFO")
    except Exception as e:
        rank0_log(f"Error loading LoRA adapter: {e}", "ERROR")
        raise


def encode_video_to_latents(pipe, video_path, max_frames=None):
    """Encode a video file into VAE latents, with resize & center crop
    to match the pipeline's resolution buckets.

    Follows the same processing flow as infer.py process_single_video:
    decord -> torch.from_numpy -> permute -> /255. -> resize/crop/normalize
    -> unsqueeze(0).transpose(1,2).to(device,dtype) -> vae.encode
    
    Args:
        pipe: The HunyuanVideo pipeline (has .vae attribute).
        video_path: Path to the video file.
        max_frames: Maximum number of frames to encode.
    
    Returns:
        latents: Tensor of shape (1, C, F', H', W') - encoded video latents.
    """
    from decord import VideoReader
    rank0_log(f"Encoding video to latents: {video_path}", "INFO")

    video_reader = VideoReader(video_path)
    ori_len = len(video_reader)
    if max_frames is not None:
        ori_len = min(ori_len, max_frames)
    if ori_len < 33:
        raise ValueError(
            f"Condition video is too short: {video_path} has {ori_len} frames (minimum 33 required)"
        )
    elif ori_len > 161:
        tgt_len = 161
        rank0_log(f"Condition video {video_path} has {ori_len} frames, truncating to {tgt_len}", "WARNING")
    else:
        tgt_len = (ori_len - 1) // 8 * 8 + 1

    batch_index = list(range(tgt_len))
    video_images = video_reader.get_batch(batch_index)

    pixel_values = torch.from_numpy(video_images.asnumpy())
    pixel_values = pixel_values.permute(0, 3, 1, 2).contiguous()
    pixel_values = pixel_values / 255.

    crop_size_list = generate_crop_size_list(base_size=640)
    ratios = np.array([round(float(h) / float(w), 5) for h, w in crop_size_list])

    height, width = pixel_values.size(-2), pixel_values.size(-1)
    aspect_ratio = float(height) / float(width)
    closest_ratio_id = np.abs(ratios - aspect_ratio).argmin()
    closest_size = crop_size_list[closest_ratio_id]

    if closest_size[0] / height > closest_size[1] / width:
        resize_size = closest_size[0], int(width * closest_size[0] / height)
    else:
        resize_size = int(height * closest_size[1] / width), closest_size[1]

    pixel_transforms = transforms.Compose([
        transforms.Resize(resize_size, antialias=True),
        transforms.CenterCrop(closest_size),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
    ])

    pixel_values = pixel_transforms(pixel_values).unsqueeze(0).transpose(1, 2).to(pipe.execution_device)

    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True), auto_offload_model(pipe.vae, pipe.execution_device, enabled=pipe.enable_offloading), pipe.vae.memory_efficient_context():
        latents = pipe.vae.encode(pixel_values).latent_dist.mode()
        # if hasattr(pipe.vae.config, "shift_factor") and pipe.vae.config.shift_factor:
        #     latents = (latents - pipe.vae.config.shift_factor) * pipe.vae.config.scaling_factor
        # else:
        #     latents = latents * pipe.vae.config.scaling_factor

    rank0_log(
        f"Encoded video latents shape: {latents.shape} from {tgt_len} frames "
        f"(resized to {closest_size[0]}x{closest_size[1]})", "INFO"
    )
    return latents


def generate_video(args):

    if int(os.environ.get('RANK', '0')) == 0:
        rank0_log("Generation arguments:", "INFO")
        for key, value in sorted(vars(args).items()):
            rank0_log(f"  {key}: {value}", "INFO")

    infer_state = initialize_infer_state(args)

    if args.sparse_attn and args.use_sageattn:
        raise ValueError("sparse_attn and use_sageattn cannot be enabled simultaneously. Please enable only one of them.")

    if args.use_fp8_gemm and 'sgl' in args.quant_type:
        try:
            import sgl_kernel
        except Exception:
            raise ValueError("sgl_kernel is not installed. Please install it using `pip install sgl-kernel==0.3.18`")
    
    task = args.task
    
    if args.dtype == 'bf16':
        transformer_dtype = torch.bfloat16
    elif args.dtype == 'fp32':
        transformer_dtype = torch.float32
    else:
        raise ValueError(f"Unsupported dtype: {args.dtype}. Must be 'bf16' or 'fp32'")
    
    # Determine offloading settings
    enable_offloading = args.offloading
    if args.group_offloading is None:
        # Auto-detect based on offloading config
        offloading_config = HunyuanVideo_1_5_Pipeline.get_offloading_config()
        enable_group_offloading = offloading_config['enable_group_offloading']
    else:
        enable_group_offloading = args.group_offloading
    
    overlap_group_offloading = args.overlap_group_offloading
    
    # Determine device and transformer_init_device based on offloading settings
    if enable_offloading:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda')
    
    if enable_group_offloading:
        transformer_init_device = torch.device('cpu')
    else:
        transformer_init_device = device
    
    pipe = HunyuanVideo_1_5_Pipeline.create_pipeline(
        pretrained_model_name_or_path=args.model_path,
        transformer_dtype=transformer_dtype,
        device=device,
        transformer_init_device=transformer_init_device,
        pipeline_config=args.pipeline_config,
    )
    
    loguru.logger.info(f"{enable_offloading=} {enable_group_offloading=} {overlap_group_offloading=}")
    
    pipe.apply_infer_optimization(
        infer_state=infer_state,
        enable_offloading=enable_offloading,
        enable_group_offloading=enable_group_offloading,
        overlap_group_offloading=overlap_group_offloading,
    )
    
    # Load checkpoint if provided
    if args.checkpoint_path:
        load_checkpoint_to_transformer(pipe, args.checkpoint_path)
    
    if args.lora_path:
        load_lora_adapter(pipe, args.lora_path)

    # ---- Prepare task-specific inputs ----
    extra_kwargs = {}
    extra_kwargs['task_type'] = task

    # Prepare ref_image_paths for tasks that need reference images
    ref_image_paths = args.ref_image_paths
    if ref_image_paths is None and args.image_path is not None:
        # Backward compatibility: convert single --image_path to list
        ref_image_paths = [args.image_path]

    if task in ('i2v',):
        if ref_image_paths is None or len(ref_image_paths) == 0:
            raise ValueError(f"Task '{task}' requires --ref_image_paths (or --image_path).")
        extra_kwargs['ref_image_paths'] = ref_image_paths

    elif task in ('interpolation', 'reference2v'):
        if ref_image_paths is None or len(ref_image_paths) == 0:
            raise ValueError(f"Task '{task}' requires --ref_image_paths with one or more image paths.")
        # interpolation / reference2v expects ref_image_paths as a list of lists
        extra_kwargs['ref_image_paths'] = [ref_image_paths]

    elif task in ('editing',):
        # editing needs condition_video_latents and condition_videos
        condition_video_latents = None
        if args.condition_video_latents_path:
            rank0_log(f"Loading pre-computed condition video latents from {args.condition_video_latents_path}", "INFO")
            condition_video_latents = torch.load(args.condition_video_latents_path, map_location='cpu')
        elif args.condition_video_paths and len(args.condition_video_paths) > 0:
            rank0_log(
                f"Task '{task}': encoding condition video to latents: "
                f"{args.condition_video_paths[0]} (max_frames={args.video_length})", "INFO",
            )
            condition_video_latents = encode_video_to_latents(
                pipe, args.condition_video_paths[0], max_frames=args.video_length
            )
        else:
            raise ValueError(f"Task '{task}' requires --condition_video_paths or --condition_video_latents_path.")
        extra_kwargs['condition_video_latents'] = condition_video_latents
        extra_kwargs['condition_videos'] = args.condition_video_paths

    elif task in ('tiv2v',):
        # tiv2v needs condition_video_latents, condition_videos, and ref_image_paths
        condition_video_latents = None
        if args.condition_video_latents_path:
            rank0_log(f"Loading pre-computed condition video latents from {args.condition_video_latents_path}", "INFO")
            condition_video_latents = torch.load(args.condition_video_latents_path, map_location='cpu')
        elif args.condition_video_paths and len(args.condition_video_paths) > 0:
            rank0_log(
                f"Task '{task}': encoding condition video to latents: "
                f"{args.condition_video_paths[0]} (max_frames={args.video_length})", "INFO",
            )
            condition_video_latents = encode_video_to_latents(
                pipe, args.condition_video_paths[0], max_frames=args.video_length
            )
        else:
            raise ValueError(f"Task '{task}' requires --condition_video_paths or --condition_video_latents_path.")
        if ref_image_paths is None or len(ref_image_paths) == 0:
            raise ValueError(f"Task '{task}' requires --ref_image_paths for condition images.")
        extra_kwargs['condition_video_latents'] = condition_video_latents
        extra_kwargs['condition_videos'] = args.condition_video_paths
        extra_kwargs['ref_image_paths'] = ref_image_paths

    elif task == 't2v':
        pass  # No extra inputs needed
    else:
        raise ValueError(f"Unknown task type: '{task}'. Supported: t2v, i2v, interpolation, reference2v, editing, tiv2v")

    out = pipe(
        prompt=args.prompt,
        aspect_ratio=args.aspect_ratio,
        num_inference_steps=args.num_inference_steps,
        video_length=args.video_length,
        negative_prompt=args.negative_prompt,
        seed=args.seed,
        think=args.think,
        output_type="pt",
        only_give_text=args.only_give_text,
        deepstack=args.deepstack,
        setclip=args.setclip,
        **extra_kwargs,
    )
    
    if int(os.environ.get('RANK', '0')) == 0:
        output_path = args.output_path
        if output_path is None:
            now = f'{datetime.datetime.now():%Y-%m-%d %H:%M:%S}'
            output_path = f'./outputs/output_{now}.mp4'
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        if args.condition_video_paths and len(args.condition_video_paths) > 0:
            from decord import VideoReader
            vr = VideoReader(args.condition_video_paths[0])
            fps = round(vr.get_avg_fps())
            del vr
        elif args.fps is not None:
            fps = args.fps
        elif args.video_length <= 81:
            fps = 16
        else:
            fps = 24
        
        save_video(out.videos, output_path, fps=fps)
        print(f"Saved video to: {output_path} (fps={fps})")
        
        if args.save_generation_config:
            try:
                save_config(args, output_path, task)
            except Exception:
                pass

def main():
    parser = argparse.ArgumentParser(description='Generate video using HunyuanVideo-1.5')

    parser.add_argument(
        '--prompt', type=str, required=True,
        help='Text prompt for video generation'
    )
    parser.add_argument(
        '--negative_prompt', type=str, default='',
        help='Negative prompt for video generation (default: empty string)'
    )
    parser.add_argument(
        '--resolution', type=str, default='480p', choices=['480p'],
        help='Video resolution'
    )
    parser.add_argument(
        '--model_path', type=str, required=True,
        help='Path to pretrained model'
    )
    parser.add_argument(
        '--pipeline_config', type=str, default='omniweaving',
        choices=list(PIPELINE_CONFIGS.keys()),
        help=f'Pipeline configuration preset (default: omniweaving). Available: {", ".join(PIPELINE_CONFIGS.keys())}'
    )
    parser.add_argument(
        '--task', type=str, default=None,
        choices=['t2v', 'i2v', 'interpolation', 'reference2v', 'editing', 'tiv2v'],
        help='Task type. If not specified, auto-detected from inputs: '
             'i2v if --image_path or --ref_image_paths is given, otherwise t2v.'
    )
    parser.add_argument(
        '--aspect_ratio', type=str, default='16:9',
        help='Aspect ratio (default: 16:9)'
    )
    parser.add_argument(
        '--num_inference_steps', type=int, default=None,
        help='Number of inference steps (default: 50)'
    )
    parser.add_argument(
        '--video_length', type=int, default=81,
        help='Number of frames to generate (default: 81)'
    )
    
    parser.add_argument(
        '--think', action='store_true', default=False,
        help='Enable AR-based prompt enhancement (think) before generation (default: false).'
    )
    parser.add_argument(
        '--sparse_attn', type=str_to_bool, nargs='?', const=True, default=False,
        help='Enable sparse attention (default: false). '
             'Use --sparse_attn or --sparse_attn true/1 to enable, '
             '--sparse_attn false/0 to disable'
    )
    parser.add_argument(
        '--offloading', type=str_to_bool, nargs='?', const=True, default=True,
        help='Enable offloading (default: true). '
             'Use --offloading or --offloading true/1 to enable, '
             '--offloading false/0 to disable'
    )
    parser.add_argument(
        '--group_offloading', type=str_to_bool, nargs='?', const=True, default=None,
        help='Enable group offloading (default: None, automatically enabled if offloading is enabled). '
             'Use --group_offloading or --group_offloading true/1 to enable, '
             '--group_offloading false/0 to disable'
    )
    parser.add_argument(
        '--overlap_group_offloading', type=str_to_bool, nargs='?', const=True, default=True,
        help='Enable overlap group offloading (default: true). '
             'Significantly increases CPU memory usage but speeds up inference. '
             'Use --overlap_group_offloading or --overlap_group_offloading true/1 to enable, '
             '--overlap_group_offloading false/0 to disable'
    )
    parser.add_argument(
        '--dtype', type=str, default='bf16', choices=['bf16', 'fp32'],
        help='Data type for transformer (default: bf16). '
             'bf16: faster, lower memory; fp32: better quality, slower, higher memory'
    )
    parser.add_argument(
        '--seed', type=int, default=123,
        help='Random seed (default: 123)'
    )
    parser.add_argument(
        '--image_path', type=str, default=None,
        help='Path to reference image for i2v (if provided and --task not set, uses i2v mode). '
             'For backward compatibility; prefer --ref_image_paths for new usage.'
    )
    parser.add_argument(
        '--ref_image_paths', type=str, nargs='+', default=None,
        help='One or more reference image paths. '
             'Required for i2v (1 image), interpolation (2+ images), reference2v (1+ images), tiv2v (1+ images).'
    )
    parser.add_argument(
        '--condition_video_paths', type=str, nargs='+', default=None,
        help='Condition video path(s) for editing/tiv2v tasks. '
             'The first video will be encoded to latents if --condition_video_latents_path is not provided.'
    )
    parser.add_argument(
        '--condition_video_latents_path', type=str, default=None,
        help='Path to pre-computed condition video latents (.pt file) for editing/tiv2v tasks. '
             'If not provided, latents will be computed from --condition_video_paths using the VAE.'
    )
    parser.add_argument(
        '--output_path', type=str, default=None,
        help='Output file path for generated video (if not provided, saves to ./outputs/output.mp4)'
    )
    parser.add_argument(
        '--fps', type=int, default=None,
        help='FPS for output video. If not specified: uses input video FPS for editing/tiv2v tasks, '
             '16 for <=81 frames, 24 for >81 frames.'
    )
    parser.add_argument(
        '--use_sageattn', type=str_to_bool, nargs='?', const=True, default=False,
        help='Enable sageattn (default: false). '
             'Use --use_sageattn or --use_sageattn true/1 to enable, '
             '--use_sageattn false/0 to disable'
    )
    parser.add_argument(
        '--sage_blocks_range', type=str, default="0-53",
        help='Sageattn blocks range (e.g., 0-5 or 0,1,2,3,4,5)'
    )
    parser.add_argument(
        '--enable_torch_compile', type=str_to_bool, nargs='?', const=True, default=False,
        help='Enable torch compile for transformer (default: false). '
             'Use --enable_torch_compile or --enable_torch_compile true/1 to enable, '
             '--enable_torch_compile false/0 to disable'
    )
    parser.add_argument(
        '--enable_cache', type=str_to_bool, nargs='?', const=True, default=False,
        help='Enable cache for transformer (default: false). '
             'Use --enable_cache or --enable_cache true/1 to enable, '
             '--enable_cache false/0 to disable'
    )
    parser.add_argument(
        '--cache_type', type=str, default="deepcache",
        help='Cache type for transformer (e.g., deepcache, teacache, taylorcache)'
    )
    parser.add_argument(
        '--no_cache_block_id', type=str, default="53",
        help='Blocks to exclude from deepcache (e.g., 0-5 or 0,1,2,3,4,5)'
    )
    parser.add_argument(
        '--cache_start_step', type=int, default=11,
        help='Start step to skip when using cache (default: 11)'
    )
    parser.add_argument(
        '--cache_end_step', type=int, default=45,
        help='End step to skip when using cache (default: 45)'
    )
    parser.add_argument(
        '--total_steps', type=int, default=50,
        help='Total inference steps (default: 50)'
    )
    parser.add_argument(
        '--cache_step_interval', type=int, default=4,
        help='Step interval to skip when using cache (default: 4)'
    )
    parser.add_argument(
        '--save_generation_config', type=str_to_bool, nargs='?', const=True, default=True,
        help='Save generation config file (default: true). '
             'Use --save_generation_config or --save_generation_config true/1 to enable, '
             '--save_generation_config false/0 to disable'
    )
    parser.add_argument(
        '--checkpoint_path', type=str, default=None,
        help='Path to checkpoint directory containing transformer weights (e.g., ./outputs/checkpoint-1000/transformer). '
             'The checkpoint directory should contain a "transformer" subdirectory. '
             'If provided, the transformer model weights will be loaded from this checkpoint.'
    )
    parser.add_argument(
        '--lora_path', type=str, default=None,
        help='Path to LoRA adapter directory or checkpoint directory containing LoRA adapter. '
             'If provided, the LoRA adapter will be loaded to the transformer model.'
    )

    parser.add_argument(
        '--only_give_text', type=str_to_bool, nargs='?', const=True, default=False,
        help='If true, the text encoder ignores visual inputs and uses text only (default: true). '
             'Set to false to enable multimodal conditioning with images/video frames in the text encoder.'
    )
    parser.add_argument(
        '--deepstack', type=int, nargs='*', default=[8, 16, 24],
        help='Deepstack layer indices for the text encoder (default: empty list). '
             'E.g., --deepstack 0 1 2'
    )
    parser.add_argument(
        '--setclip', type=str_to_bool, nargs='?', const=True, default=True,
        help='Enable CLIP-based features in the text encoder (default: false).'
    )

    # fp8 gemm related
    parser.add_argument(
        '--use_fp8_gemm', type=str_to_bool, nargs='?', const=True, default=False,
        help='Enable fp8 gemm for transformer (default: false). '
             'Use --use_fp8_gemm or --use_fp8_gemm true/1 to enable, '
             '--use_fp8_gemm false/0 to disable'
    )
    parser.add_argument(
        '--quant_type', type=str, default="fp8-per-token-sgl",
        help='Quantization type for fp8 gemm (e.g., fp8-per-tensor-weight-only, fp8-per-tensor, fp8-per-token-sgl)'
    )
    parser.add_argument(
        '--include_patterns', type=str, default="double_blocks",
        help='Include patterns for fp8 gemm (default: double_blocks)'
    )

    args = parser.parse_args()
    
    # Convert string "none" to None for image_path
    if args.image_path is not None and args.image_path.lower().strip() == 'none':
        args.image_path = None
    
    # Auto-detect task type if not explicitly specified
    if args.task is None:
        if args.condition_video_paths and args.ref_image_paths:
            args.task = 'tiv2v'
        elif args.condition_video_paths or args.condition_video_latents_path:
            args.task = 'editing'
        elif args.ref_image_paths and len(args.ref_image_paths) >= 2:
            args.task = 'interpolation'
        elif args.image_path or args.ref_image_paths:
            args.task = 'i2v'
        else:
            args.task = 't2v'
        rank0_log(f"Auto-detected task type: {args.task}", "INFO")
    
    generate_video(args)
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
