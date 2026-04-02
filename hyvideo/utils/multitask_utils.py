import torch
import math
import random
import torchvision.transforms as transforms
from typing import List
import numpy as np
import PIL.Image
from PIL import Image
from decord import VideoReader

def numpy_to_pil(images: np.ndarray) -> List[PIL.Image.Image]:
    r"""
    Convert a numpy image or a batch of images to a PIL image.

    Args:
        images (`np.ndarray`):
            The image array to convert to PIL format.

    Returns:
        `List[PIL.Image.Image]`:
            A list of PIL images.
    """
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    if images.shape[-1] == 1:
        # special case for grayscale (single channel) images
        pil_images = [Image.fromarray(image.squeeze(), mode="L") for image in images]
    else:
        pil_images = [Image.fromarray(image) for image in images]

    return pil_images

def black_image(width, height):
    black_image = Image.new('RGB', (width, height), (0, 0, 0))
    return black_image

def get_mask_from_mask_type(video_length, mask_type, mask_args, seed=None, num_imgs=[1]):

    if mask_type == "t2v":
        mask = torch.zeros(video_length)
    elif mask_type == "frame_interpolation":
        frame_interpolation_rate = mask_args["frame_interpolation_rate"]
        a = video_length // frame_interpolation_rate + 1
        mask = [torch.ones(a)]
        for i in range(frame_interpolation_rate - 1):
            mask.append(torch.zeros(a))
        mask = torch.stack(mask, dim=-1).reshape(-1)[:video_length]
    elif mask_type == "image_interpolation":
        mask = torch.zeros(video_length)
        mask[0] = 1.0
        mask[-1] = 1.0
    elif mask_type == "video_connect":
        mask = torch.cat((torch.ones(video_length//4), torch.zeros(video_length-((video_length//4)*2)), torch.ones(video_length//4)))
    elif mask_type == "random":
        mask = torch.randint(0, 2, (video_length,)).float()
    elif mask_type == "expansion":
        expansion_save_rate = mask_args["expansion_save_rate"]
        expansion_reverse_rate = mask_args["expansion_reverse_rate"]
        if seed is not None:
            rng = random.Random(seed + 1)
            random_num = rng.random()
        else:
            random_num = random.random() # [0, 1)
        if random_num < expansion_reverse_rate:
            expansion_reverse = True
        else:
            expansion_reverse = False

        video_length_save = math.floor(video_length * expansion_save_rate)
        video_length_drop = video_length - video_length_save
        assert video_length_drop != 0, f"video_length_drop: {video_length_drop} must be non-zero."
        mask_save = torch.ones((video_length_save))
        mask_drop = torch.zeros((video_length_drop))
        if expansion_reverse:
            mask = torch.concat([mask_drop, mask_save], dim=-1)
        else:
            mask = torch.concat([mask_save, mask_drop], dim=-1)
    elif mask_type == "i2v":
        mask = torch.zeros(video_length)
        mask[0] = 1.0
    elif mask_type == 'reference2v':
        B = len(num_imgs)
        mask = torch.zeros(B, video_length)
        for bb, num in enumerate(num_imgs):
            mask[bb, 1:num+1] = 1.0
        if B==1:
            mask = mask.squeeze(0)
    elif mask_type == 'editing':
        mask = torch.ones(video_length)
    elif mask_type == 'tiv2v':
        mask1 = torch.ones(video_length)
        mask2 = torch.zeros(video_length)
        mask2[1] = 1.0
        mask = torch.concat([mask1, mask2], dim=-1)
    elif mask_type == 'interpolation':
        mask = torch.zeros(video_length)
        mask[0] = 1.0
        mask[-1] = 1.0
    else:
        raise ValueError(f"{mask_type} is not supported !")

    return mask

def multitask_check(multi_task_args):
    count_prob = 0
    for k, v in multi_task_args.items():
        assert k in ["t2v", "frame_interpolation", "image_interpolation", "expansion", "i2v", "video_connect", "random"], \
            f"{k} must be within [t2v, frame_interpolation, image_interpolation, expansion, i2v, video_connect, random]"
        count_prob += v["prob"]
        if k == "frame_interpolation":
            assert "mask_args" in v, f"mask_args must be provided for frame_interpolation"
            assert "frame_interpolation_rate" in v["mask_args"], f"frame_interpolation_rate must be provided for frame_interpolation"
        elif k == "expansion":
            assert "mask_args" in v, f"mask_args must be provided for expansion"
            assert "expansion_save_rate" in v["mask_args"], f"expansion_save_rate must be provided for expansion"
            assert "expansion_reverse_rate" in v["mask_args"], f"expansion_reverse_rate must be provided for expansion"
    assert abs(1 - count_prob) < 0.000001, f"sum(prob): {count_prob} must be 1"

def get_multitask_mask(video_length, multi_task_args, seed=None):
    cumsum_threshold = 0
    thresholds = [0]
    mask_types = []
    for k, v in multi_task_args.items():
        cumsum_threshold += v["prob"]
        thresholds.append(cumsum_threshold)
        mask_types.append(k)

    random_num = random.random()
    
    for i in range(0, len(thresholds) - 1):
        if thresholds[i] <= random_num < thresholds[i + 1]:
            mask_type = mask_types[i]
            if mask_type == "frame_interpolation" or mask_type == "expansion":
                mask_args = multi_task_args[mask_type]["mask_args"]
            else:
                mask_args = None
            break
    if video_length == 1:
        mask_type = "t2v"
        mask_args = None
    multi_task_mask = get_mask_from_mask_type(video_length, mask_type=mask_type, mask_args=mask_args, seed=seed)
    return multi_task_mask, mask_type

def get_multitask_mask_reference2v(video_length, seed=None, num_imgs=[1]):
    if video_length == 1:
        mask_type = "t2v"
        mask_args = None
    else:
        mask_type = 'reference2v'
        multi_task_mask = get_mask_from_mask_type(video_length, mask_type=mask_type, mask_args=None, seed=seed, num_imgs=num_imgs)
    return multi_task_mask, mask_type

def get_multitask_mask_i2v(video_length, seed=None):
    if video_length == 1:
        mask_type = "t2v"
        mask_args = None
    else:
        mask_type = 'i2v'
        multi_task_mask = get_mask_from_mask_type(video_length, mask_type=mask_type, mask_args=None, seed=seed)
    return multi_task_mask, mask_type

def get_multitask_mask_t2v(video_length, seed=None):
    mask_type = 't2v'
    multi_task_mask = get_mask_from_mask_type(video_length, mask_type=mask_type, mask_args=None, seed=seed)
    return multi_task_mask, mask_type

def get_multitask_mask_editing(video_length, seed=None):
    if video_length == 1:
        mask_type = "t2v"
        mask_args = None
    else:
        mask_type = 'editing'
        multi_task_mask = get_mask_from_mask_type(video_length, mask_type=mask_type, mask_args=None, seed=seed)
    return multi_task_mask, mask_type

def get_multitask_mask_tiv2v(video_length, seed=None):
    if video_length == 1:
        mask_type = "t2v"
        mask_args = None
    else:
        mask_type = 'tiv2v'
        multi_task_mask = get_mask_from_mask_type(video_length, mask_type=mask_type, mask_args=None, seed=seed)
    return multi_task_mask, mask_type

def get_multitask_mask_interpolation(video_length, seed=None):
    if video_length == 1:
        mask_type = "t2v"
        mask_args = None
    else:
        mask_type = 'interpolation'
        multi_task_mask = get_mask_from_mask_type(video_length, mask_type=mask_type, mask_args=None, seed=seed)
    return multi_task_mask, mask_type

def merge_tensor_by_mask_batched(tensor_1, tensor_2, mask, dim):
    """
    Args:
        tensor_1, tensor_2: [B, C, F, H, W]
        mask: [B, F] or a mask matching the tensor along `dim`
        dim: dimension to apply the mask on (e.g. dim=2 corresponds to F)
    """
    ndims = tensor_1.dim()

    view_shape = [1] * ndims
    view_shape[0] = mask.shape[0]
    view_shape[dim] = mask.shape[1]

    mask_reshaped = mask.view(*view_shape)

    return torch.where(mask_reshaped == 1, tensor_2, tensor_1)

def merge_tensor_by_mask(tensor_1, tensor_2, mask, dim):
    assert tensor_1.shape == tensor_2.shape
    # Mask is a 0/1 vector. Choose tensor_2 when the value is 1; otherwise, tensor_1
    masked_indices = torch.nonzero(mask).squeeze(1)
    tmp = tensor_1.clone()
    if dim == 0:
        tmp[masked_indices] = tensor_2[masked_indices]
    elif dim == 1:
        tmp[:, masked_indices] = tensor_2[:, masked_indices]
    elif dim == 2:
        tmp[:, :, masked_indices] = tensor_2[:, :, masked_indices]
    return tmp

def get_cond_latents(latents, vae, is_mode=True):
    first_image_latents = latents[:, :, 0, ...] if len(latents.shape) == 5 else latents
    first_image_latents = 1 / vae.config.scaling_factor * first_image_latents
    first_images = vae.decode(first_image_latents.unsqueeze(2).to(vae.dtype), return_dict=False)[0]
    first_images = first_images.squeeze(2)
    first_images = (first_images / 2 + 0.5).clamp(0, 1)
    first_images = first_images.cpu().permute(0, 2, 3, 1).float().numpy()
    first_images_pil = numpy_to_pil(first_images)
    first_images_np = (first_images * 255).round().astype("uint8")
    
    image_transform = transforms.Compose([
            transforms.ToTensor(), 
            transforms.Normalize([0.5], [0.5])
        ])
    first_images_pixel_values = [image_transform(image).unsqueeze(0) for image in first_images_pil]
    first_images_pixel_values = torch.cat(first_images_pixel_values).unsqueeze(2).to(vae.device)
    
    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
        if is_mode:
            cond_latents = vae.encode(first_images_pixel_values).latent_dist.mode()  # B, C, F, H, W
        else:
            cond_latents = vae.encode(first_images_pixel_values).latent_dist.sample() # B, C, F, H, W
        cond_latents.mul_(vae.config.scaling_factor)
        
        
    return cond_latents, first_images_np, first_images_pil

# subject_driven
def get_cond_latents2(first_images_pil, vae, F, is_mode=True):
    """
    Get condition latents from nested PIL Image list.
    
    Args:
        args: Arguments object containing vae_precision
        first_images_pil: List[List[PIL.Image.Image]], where each inner list contains 1-4 images
        vae: VAE model
        F: Time dimension (must be > 5)
        is_uncond: If True, replace images with black images
        is_mode: If True, use mode() for latent distribution, else use sample()
    
    Returns:
        cond_latents: Encoded latents tensor (B, C, F, H, W)
            - For each batch b, if first_images_pil[b] has N images (1-4),
              cond_latents[b, :, :N, :, :] contains encoded values,
              cond_latents[b, :, N:, :, :] are zeros
        first_images_np: List of numpy arrays of images (uint8) for each batch
        first_images_pil: List of lists of PIL images (may be replaced with black images if is_uncond)
    """
    
    B = len(first_images_pil)
    
    # Prepare image transform
    image_transform = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize([0.5], [0.5])
    ])
    
    # Process each batch
    batch_cond_latents_list = []
    batch_images_np_list = []
    batch_images_pil_list = []
    
    # Get first image dimensions to determine latent shape (assuming all images same size)
    first_img = first_images_pil[0][0] if first_images_pil[0] else None
    if first_img is None:
        raise ValueError("first_images_pil cannot contain empty lists")
    
    processed_images_pil = first_images_pil
    
    
    for batch_idx, batch_images in enumerate(processed_images_pil):
        num_images = len(batch_images)
        assert 1 <= num_images <= 4, f"Each batch must have 1-4 images, got {num_images} for batch {batch_idx}"
        
        batch_images_np = np.array([np.array(img, dtype=np.uint8) for img in batch_images])
        # PIL Image values are already in [0, 255] range, so no need to multiply by 255
        # batch_images_np = np.array([np.array(img, dtype=np.uint8) for img in batch_images])
        batch_images_np_list.append(batch_images_np)
        batch_images_pil_list.append(batch_images.copy())
        
        # Transform PIL images to tensor format
        batch_pixel_values = [image_transform(img).unsqueeze(0) for img in batch_images]
        batch_pixel_values = torch.cat(batch_pixel_values).unsqueeze(2).to(vae.device)  # (num_images, C, 1, H, W)
        
        # Encode images to latents
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
            if is_mode:
                encoded_latents = vae.encode(batch_pixel_values).latent_dist.mode()  # (num_images, C, 1, H, W)
            else:
                encoded_latents = vae.encode(batch_pixel_values).latent_dist.sample()  # (num_images, C, 1, H, W)
            encoded_latents.mul_(vae.config.scaling_factor)
        
        # Get latent dimensions (C, H, W)
        C, H, W = encoded_latents.shape[1], encoded_latents.shape[3], encoded_latents.shape[4]
        
        # Create full latent tensor for this batch: (1, C, F, H, W) with zeros
        batch_cond_latents = torch.zeros(1, C, F, H, W, dtype=encoded_latents.dtype, device=encoded_latents.device)
        
        # Fill first num_images frames with encoded values
        # encoded_latents: (num_images, C, 1, H, W) -> squeeze(2) -> (num_images, C, H, W) -> unsqueeze(0) -> (1, num_images, C, H, W) -> permute(0,2,1,3,4) -> (1, C, num_images, H, W)
        encoded_values = encoded_latents.squeeze(2)  # (num_images, C, H, W)
        batch_cond_latents[0, :, 1:num_images+1, :, :] = encoded_values.permute(1, 0, 2, 3)  # (C, num_images, H, W) -> (1, C, num_images, H, W)
        
        batch_cond_latents_list.append(batch_cond_latents)
    
    # Concatenate all batches
    cond_latents = torch.cat(batch_cond_latents_list, dim=0)  # (B, C, F, H, W)
    
    return cond_latents, batch_images_np_list, batch_images_pil_list

def get_cond_latents3(first_images_pil, vae, F, is_mode=True):
    """
    Get condition latents from nested PIL Image list.
    
    Args:
        args: Arguments object containing vae_precision
        first_images_pil: List[List[PIL.Image.Image]], where each inner list contains exactly 2 images
        vae: VAE model
        F: Time dimension (must be > 0)
        is_uncond: If True, replace images with black images
        is_mode: If True, use mode() for latent distribution, else use sample()
    
    Returns:
        cond_latents: Encoded latents tensor (B, C, F, H, W)
            - For each batch b, first_images_pil[b] has exactly 2 images,
              cond_latents[b, :, 0, :, :] contains the first encoded image,
              cond_latents[b, :, F-1, :, :] contains the second encoded image,
              cond_latents[b, :, 1:F-1, :, :] are zeros
        first_images_np: List of numpy arrays of images (uint8) for each batch
        first_images_pil: List of lists of PIL images (may be replaced with black images if is_uncond)
    """
    
    B = len(first_images_pil)
    
    # Prepare image transform
    image_transform = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize([0.5], [0.5])
    ])
    
    # Process each batch
    batch_cond_latents_list = []
    batch_images_np_list = []
    batch_images_pil_list = []
    
    # Get first image dimensions to determine latent shape (assuming all images same size)
    first_img = first_images_pil[0][0] if first_images_pil[0] else None
    if first_img is None:
        raise ValueError("first_images_pil cannot contain empty lists")
    
    processed_images_pil = first_images_pil
    
    for batch_idx, batch_images in enumerate(processed_images_pil):
        num_images = len(batch_images)
        assert num_images == 2, f"Each batch must have exactly 2 images, got {num_images} for batch {batch_idx}"
        
        # Convert images to numpy for return
        batch_images_np = np.array([np.array(img, dtype=np.uint8) for img in batch_images])
        batch_images_np_list.append(batch_images_np)
        batch_images_pil_list.append(batch_images.copy())
        
        # Transform PIL images to tensor format
        batch_pixel_values = [image_transform(img).unsqueeze(0) for img in batch_images]
        batch_pixel_values = torch.cat(batch_pixel_values).unsqueeze(2).to(vae.device)  # (2, C, 1, H, W)
        
        # Encode images to latents
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
            if is_mode:
                encoded_latents = vae.encode(batch_pixel_values).latent_dist.mode()  # (2, C, 1, H, W)
            else:
                encoded_latents = vae.encode(batch_pixel_values).latent_dist.sample()  # (2, C, 1, H, W)
            encoded_latents.mul_(vae.config.scaling_factor)
        
        # Get latent dimensions (C, H, W)
        C, H, W = encoded_latents.shape[1], encoded_latents.shape[3], encoded_latents.shape[4]
        
        # Create full latent tensor for this batch: (1, C, F, H, W) with zeros
        batch_cond_latents = torch.zeros(1, C, F, H, W, dtype=encoded_latents.dtype, device=encoded_latents.device)
        
        # Fill first image at position 0 and second image at position F-1
        # encoded_latents: (2, C, 1, H, W) -> squeeze(2) -> (2, C, H, W)
        encoded_values = encoded_latents.squeeze(2)  # (2, C, H, W)
        
        # First image at position 0
        batch_cond_latents[0, :, 0, :, :] = encoded_values[0]  # (C, H, W)
        
        # Second image at position F-1
        batch_cond_latents[0, :, F-1, :, :] = encoded_values[1]  # (C, H, W)
        
        batch_cond_latents_list.append(batch_cond_latents)
    
    # Concatenate all batches
    cond_latents = torch.cat(batch_cond_latents_list, dim=0)  # (B, C, F, H, W)
    
    return cond_latents, batch_images_np_list, batch_images_pil_list

def get_cond_images(latents, vae):
    sematic_image_latents = latents[:, :, 0, ...] if len(latents.shape) == 5 else latents
    sematic_image_latents = 1 / vae.config.scaling_factor * sematic_image_latents
    semantic_images = vae.decode(sematic_image_latents.unsqueeze(2).to(vae.dtype), return_dict=False)[0]
    semantic_images = semantic_images.squeeze(2)
    semantic_images = (semantic_images / 2 + 0.5).clamp(0, 1)
    semantic_images = semantic_images.cpu().permute(0, 2, 3, 1).float().numpy()
    semantic_images = numpy_to_pil(semantic_images)
    
        
    return semantic_images


def prepare_custom_video(vr, total_subset=161, nframes=8):
    if isinstance(vr, str):
        vr = VideoReader(vr)
    
    max_frames = min(len(vr), total_subset)
    sample_indices = np.linspace(0, max_frames - 1, nframes).astype(int)
    
    frames = vr.get_batch(sample_indices)
    if hasattr(frames, 'asnumpy'):
        frames = frames.asnumpy()  # decord NDArray
    else:
        frames = frames.numpy()  # PyTorch Tensor
    frame_list = [Image.fromarray(frame) for frame in frames]
    
    return frame_list

def get_semantic_images_np(video_path_list, nframes=8):
    """
    Extract first frame from each video in video_path_list and return batchified numpy array.
    
    Args:
        video_path_list: List of video paths
        
    Returns:
        first_images_np: numpy array of shape (B, H, W, C) with dtype uint8 (0-255)
            Returns None if video_path_list is None or empty
    """
    if video_path_list is None or len(video_path_list) == 0:
        return None
    
    first_frames = []
    sampled_frames = []
    for video_path in video_path_list:
        # Read first frame from video
        if isinstance(video_path, str):
            video_reader = VideoReader(video_path)
        else:
            assert isinstance(video_path, VideoReader), f"video_path must be either str or VideoReader, got {type(video_path)}"
            video_reader = video_path
        if len(video_reader) == 0:
            raise ValueError(f"Video {video_path} has no frames")
        
        sample_frame_list = prepare_custom_video(video_reader, nframes=nframes)
        sampled_frames.append(sample_frame_list)
        # Get first frame (index 0)
        first_frame = video_reader.get_batch([0])
        # Convert to numpy: shape is (1, H, W, C)
        if hasattr(first_frame, 'asnumpy'):
            first_frame_np = first_frame.asnumpy()  # decord NDArray
        else:
            first_frame_np = first_frame.numpy()  # PyTorch Tensor
        # Remove batch dimension: shape becomes (H, W, C)
        first_frame_np = first_frame_np[0]
        first_frames.append(first_frame_np)
        del video_reader
    
    # Stack all first frames into batch: shape becomes (B, H, W, C)
    first_images_np = np.stack(first_frames, axis=0)
    # Ensure uint8 format (0-255)
    first_images_np = first_images_np.astype(np.uint8)
    
    return first_images_np, sampled_frames

def get_semantic_images_np2(video_path_list, first_image_pil, vae, F, nframes=8, is_mode=True):
    """
    Extract first frame from each video in video_path_list and return batchified numpy array.
    Also encode first_image_pil list into cond_latents with values filled at F index=1.

    Args:
        video_path_list: List of video paths or VideoReader objects
        first_image_pil: List of PIL.Image.Image
        args: Arguments object containing vae_precision
        vae: VAE model
        F: Time dimension for cond_latents (must be >= 2)
        nframes: Number of frames to sample for each video (default: 8)
        is_mode: If True, use mode() for latent distribution, else use sample()

    Returns:
        first_images_np: numpy array of shape (B, H, W, C) with dtype uint8 (0-255)
        sampled_frames: list of sampled frame lists per video
        cond_latents: tensor of shape (B, C, F, H, W), values at index=1, others 0
        first_image_pil_np: numpy array of shape (B, H, W, C) with dtype uint8 (0-255)
    """
    first_images_np, sampled_frames = get_semantic_images_np(video_path_list, nframes=nframes)

    if first_image_pil is None or len(first_image_pil) == 0:
        raise ValueError("first_image_pil must be a non-empty list of PIL images")

    if F < 2:
        raise ValueError(f"F must be >= 2 to fill index=1, got {F}")

    first_image_pil_np = np.array([np.array(img, dtype=np.uint8) for img in first_image_pil])

    image_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    first_images_pixel_values = [image_transform(image).unsqueeze(0) for image in first_image_pil]
    first_images_pixel_values = torch.cat(first_images_pixel_values).unsqueeze(2).to(vae.device)

    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
        if is_mode:
            encoded_latents = vae.encode(first_images_pixel_values).latent_dist.mode()
        else:
            encoded_latents = vae.encode(first_images_pixel_values).latent_dist.sample()
        encoded_latents.mul_(vae.config.scaling_factor)

    B = encoded_latents.shape[0]
    C, H, W = encoded_latents.shape[1], encoded_latents.shape[3], encoded_latents.shape[4]
    cond_latents = torch.zeros(B, C, F, H, W, dtype=encoded_latents.dtype, device=encoded_latents.device)
    cond_latents[:, :, 1, :, :] = encoded_latents.squeeze(2)

    return first_images_np, sampled_frames, cond_latents, first_image_pil_np

if __name__ == '__main__':
    video_length = 129
    multi_task_args = {'random': {'prob': 1.0}}

    multitask_mask, mask_type = get_multitask_mask(video_length, multi_task_args, seed=42)
    print('multitask_mask = ', multitask_mask)
    print('mask_type = ', mask_type)
