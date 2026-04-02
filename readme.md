# OmniWeaving

<div align="center">

<img src="./assets/logo.png" alt="OmniWeaving Logo" width="80%">

# <img src="./assets/weaving-mark.svg" alt="icon" height="30"> OmniWeaving: Towards Unified Video Generation with Free-form Composition and Reasoning

</div>


<div align="center">

[Kaihang Pan](https://scholar.google.com/citations?user=lMQADDUAAAAJ)<sup>&ast;1,2</sup>,
[Qi Tian](https://scholar.google.com/citations?user=Ypu45nIAAAAJ)<sup>&ast;2</sup>,
[Jianwei Zhang](https://scholar.google.com/citations?user=nF_klRIAAAAJ)<sup>2</sup>,
[Weijie Kong](https://scholar.google.com/citations?user=gsOklKAAAAAJ)<sup>2</sup>,
[Jiangfeng Xiong](https://scholar.google.com/citations?user=lHbXg_0AAAAJ)<sup>2</sup>,
[Yanxin Long](https://scholar.google.com/citations?user=uqiNUvwAAAAJ)<sup>2</sup>,
[Shixue Zhang](https://scholar.google.com/citations?user=N8jMnXEAAAAJ)<sup>2</sup>,
Haiyi Qiu<sup>1</sup>,
Tan Wang<sup>3</sup>,
Zheqi Lv<sup>1</sup>,
[Yue Wu](https://scholar.google.com/citations?user=1xTR6qoAAAAJ)<sup>&sect;2</sup>,
[Liefeng Bo](https://scholar.google.com/citations?user=FJwtMf0AAAAJ)<sup>2</sup>,
[Siliang Tang](https://scholar.google.com/citations?user=8e7H3PcAAAAJ)<sup>&sect;1</sup>,
[Zhao Zhong](https://scholar.google.com/citations?user=igtXP_kAAAAJ)<sup>&dagger;2</sup>

<sup>1</sup>Zhejiang University &nbsp; <sup>2</sup>Tencent Hunyuan &nbsp; <sup>3</sup>Nanyang Technological University  
<sup>&ast;</sup>Equal Contribution &nbsp; <sup>&sect;</sup>Corresponding Authors &nbsp; <sup>&dagger;</sup>Project Leader  
Work done during Kaihang Pan's internship at Tencent Hunyuan

</div>




<div align="center">


</div>



<div align="center">
  <a href="https://omniweaving.github.io/" target="_blank"><img src=https://img.shields.io/badge/Official%20Site-333399.svg?logo=homepage height=22px></a>
  <a href=https://huggingface.co/tencent/HY-OmniWeaving target="_blank"><img src=https://img.shields.io/badge/%F0%9F%A4%97%20Models-d96902.svg height=22px></a>
  <a href=https://github.com/Tencent-Hunyuan/OmniWeaving target="_blank"><img src= https://img.shields.io/badge/GitHub-bb8a2e.svg?logo=github height=22px></a>
  <a href="https://arxiv.org/abs/2603.24458" target="_blank"><img src=https://img.shields.io/badge/Paper-b5212f.svg?logo=arxiv height=22px></a>
  <a href="https://huggingface.co/datasets/midbee/IntelligentVBench" target="_blank"><img src=https://img.shields.io/badge/%F0%9F%A4%97%20IntelligentVBench-4e72b8.svg height=22px></a>

</div>


<a id="news"></a>

## 🔥🔥🔥 News
* 📌 OmniWeaving is developed by the **HunyuanVideo** team and is built upon the latest **[HunyuanVideo-1.5](https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5)** as the backbone. If you find our work useful, please consider giving this repository a star and citing our paper~
* 🚀 April 3, 2026: We release the [code](https://github.com/Tencent-Hunyuan/OmniWeaving) 
and [model weights](https://huggingface.co/tencent/HY-OmniWeaving) of OmniWeaving.
* 🚀 April 3, 2026: We release the [IntelligentVBench](https://huggingface.co/datasets/midbee/IntelligentVBench).
* 📖 Mar 26, 2026: We release the OmniWeaving paper on [Arxiv](https://arxiv.org/abs/2603.24458).
* 👋 Mar 25, 2026: We release the webpage of [OmniWeaving](https://omniweaving.github.io/).


<a id="open-source-plan"></a>

## 📑 Open-source Plan
- **OmniWeaving**
  - [✅] Inference Code
  - [✅] Model Checkpoints
  - [✅] Training Data Construction Code
  - [TODO] Training Example Code
- **IntelligentVBench**
  - [✅] Test cases
  - [✅] Evaluation Code
  - [TODO] Reference Inference Scripts
  
## 📋 Table of Contents
- [🔥🔥🔥 News](#news)
- [📑 Open-source Plan](#open-source-plan)
- [📖 Abstract](#abstract)
- [🏗 Model Architecture](#model-architecture)
- [🚀 Supported Tasks](#supported-tasks)
- [🛠 Preparation](#preparation)
- [🔑 Inference](#inference)
- [🗂 Training Data Construction](#training-data-construction)
- [📊 Evaluation on IntelligentVBench](#evaluation-on-intelligentvbench)
- [🎬 Examples](#examples)
- [📚 Citation](#citation)
- [🙏 Acknowledgements](#acknowledgements)


<a id="abstract"></a>

## 📖 Abstract
While proprietary systems such as Seedance-2.0 have achieved remarkable success in omni-capable video generation, open-source alternatives significantly lag behind. To bridge this gap, we propose <img src="./assets/weaving-mark.svg" alt="OmniWeaving" height="18px"> **OmniWeaving**, an omni-level video generation model featuring powerful multimodal composition and reasoning-informed capabilities. By leveraging a massive-scale pretraining dataset that encompasses diverse compositional and reasoning-augmented scenarios, OmniWeaving learns to temporally bind interleaved text, multi-image, and video inputs while acting as an intelligent agent to infer complex user intentions for sophisticated video creation. 
Furthermore, we introduce **IntelligentVBench**, the first comprehensive benchmark designed to rigorously assess next-level intelligent unified video generation. Extensive experiments demonstrate that OmniWeaving achieves SoTA performance among open-source unified models.


<a id="model-architecture"></a>

## 🏗 Model Architecture

Following the paper, **OmniWeaving** is built as an integrated **MLLM + MMDiT + VAE** framework for unified free-form video generation. The **MLLM** serves as the semantic parser for interleaved text, images, and video inputs, mapping them into a high-level semantic space and forwarding its hidden states through an MLP connector. The **VAE** acts as the visual tokenizer, compressing visual inputs into low-level latents, while the **MMDiT** uses these semantic conditions together with latent noise to generate semantically aligned, high-fidelity videos.

On this basis, we further introduce two extra improvements tailored for advanced reasoning and composition.

- **(1) Activating Thinking Mode of the MLLM:** Direct MLLM encoding of interleaved visual-text inputs often yields semantic ambiguity due to weak intra-correlations and unclear video creation intents. We elevate the MLLM from a passive feature extractor to an active reasoner. By activating the thinking mode to generate intermediate reasoning steps, it autonomously deduces a semantically precise, enhanced prompt. The hidden states of this enhanced prompt are then forwarded alongside the original MLLM features to condition the MMDiT, effectively bridging the cognitive gap between abstract user intent and pixel-level generation.
- **(2) Hidden States DeepStacking:** Compositional video generation involving multiple subjects or intricate scenes often relies on both low- and high-level semantic representations. Drawing inspiration from the DeepStacking mechanism in Qwen3-VL, we extract hidden states from a broader range of intermediate MLLM layers to capture a rich semantic spectrum spanning from fine-grained details to high-level abstractions. An MLP connector projects these multi-level features into the MMDiT embedding space. These projected features are then directly added to the corresponding hidden states within the first three layers of the MMDiT conditioning branch, effectively injecting multi-granular semantic guidance into the generative process.

<div align="center">
<img src="./assets/architecture.jpg" alt="OmniWeaving Architecture" width="800">

**Figure 1.** Overview of the OmniWeaving architecture, which consists of an MLLM for multimodal understanding and an MMDiT for generation.
</div>

<a id="supported-tasks"></a>

## 🚀 Supported Tasks

OmniWeaving is flexible in its input and output configurations, supporting a wide range of unified video generation tasks:

<table>
  <thead>
    <tr>
      <th>Task</th>
      <th>Input Type</th>
      <th>Output</th>
      <th>Description</th>
      <th>Demo Input</th>
      <th>Demo Output</th>
    </tr>
  </thead>
  <tbody>
    <!-- Text-to-Video -->
    <tr>
      <td><b>Text-to-Video (T2V)</b></td>
      <td>Text 📝</td>
      <td>Video 🎬</td>
      <td>Generating a video from text prompts.</td>
      <td align="center">
        <img src="assets/prompt-badge.svg" height="18"/>
      </td>
      <td align="center">
        <img src="assets/cases/t2v/1.gif" width="140"/>
      </td>
    </tr>
    <!-- First-Frame-to-Video (I2V) -->
    <tr>
      <td><b>First-Frame-to-Video (I2V)</b></td>
      <td>Image 🖼 + Text 📝</td>
      <td>Video 🎬</td>
      <td>Generating a video based on the first frame.</td>
      <td align="center">
        <img src="assets/cases/i2v/1.png" width="140"/><br/>
        <img src="assets/prompt-badge.svg" height="18"/>
      </td>
      <td align="center">
        <img src="assets/cases/i2v/1.gif" width="140"/>
      </td>
    </tr>
    <!-- Key-Frames-to-Video -->
    <tr>
      <td><b>Key-Frames-to-Video</b></td>
      <td>2 × Images 🖼 + Text 📝</td>
      <td>Video 🎬</td>
      <td>Generating a video conditioned on start and end frames.</td>
      <td align="center">
        <img src="assets/cases/interpolation/1_first.png" width="60"/>
        <img src="assets/cases/interpolation/1_last.png" width="60"/><br/>
        <img src="assets/prompt-badge.svg" height="18"/>
      </td>
      <td align="center">
        <img src="assets/cases/interpolation/1.gif" width="140"/>
      </td>
    </tr>
    <!-- Video-to-Video Editing -->
    <tr>
      <td><b>Video-to-Video Editing</b></td>
      <td>Video 🎬 + Text 📝</td>
      <td>Video 🎬</td>
      <td>Instruction-based video manipulation and stylization.</td>
      <td align="center">
        <img src="assets/cases/editing/1_before.gif" width="140"/><br/>
        <img src="assets/prompt-badge.svg" height="18"/>
      </td>
      <td align="center">
        <img src="assets/cases/editing/1_after.gif" width="140"/>
      </td>
    </tr>
    <!-- Reference-to-Video (single image) -->
    <tr>
      <td><b>Reference-to-Video</b></td>
      <td>Image 🖼 + Text 📝</td>
      <td>Video 🎬</td>
      <td>Single-subject reference-driven video generation.</td>
      <td align="center">
        <img src="assets/cases/reference2v/1.png" width="100"/><br/>
        <img src="assets/prompt-badge.svg" height="18"/>
      </td>
      <td align="center">
        <img src="assets/cases/reference2v/1.gif" width="140"/>
      </td>
    </tr>
    <!-- Compositional Multi-Image-to-Video (multiple images) -->
    <tr>
      <td><b>Compositional Multi-Image-to-Video</b></td>
      <td>2–4 × Images 🖼 + Text 📝</td>
      <td>Video 🎬</td>
      <td>Multi-subject compositional video generation.</td>
      <td align="center">
        <img src="assets/cases/compositional/1_1.png" width="40"/>
        <img src="assets/cases/compositional/1_2.png" width="40"/>
        <img src="assets/cases/compositional/1_3.png" width="40"/><br/>
        <img src="assets/prompt-badge.svg" height="18"/>
      </td>
      <td align="center">
        <img src="assets/cases/compositional/1.gif" width="140"/>
      </td>
    </tr>
    <!-- Text-Image-Video-to-Video -->
    <tr>
      <td><b>Text-Image-Video-to-Video</b></td>
      <td>Video 🎬 + Image 🖼 + Text 📝</td>
      <td>Video 🎬</td>
      <td>Generating a video conditioned on text, image, and video inputs.</td>
      <td align="center">
        <img src="assets/cases/tiv2v/1_ref.png" width="50"/><br/>
        <img src="assets/cases/tiv2v/1_before.gif" width="140"/><br/>
        <img src="assets/prompt-badge.svg" height="18"/>
      </td>
      <td align="center">
        <img src="assets/cases/tiv2v/1_after.gif" width="140"/>
      </td>
    </tr>
    <!-- Reasoning-Augmented Video Generation -->
    <tr>
      <td><b>Reasoning-Augmented Video Generation</b></td>
      <td>Image(s) 🖼 + Text 📝</td>
      <td>Reasoning 💭 + Video 🎬</td>
      <td>Reasoning over user intent before generating the video.</td>
      <td align="center">
        <img src="assets/cases/reasoning/1.png" width="140"/><br/>
        <img src="assets/prompt-badge.svg" height="18"/>
      </td>
      <td align="center">
        <img src="assets/reasoning-badge.svg" height="18"/><br/>
        <img src="assets/cases/reasoning/1.gif" width="140"/>
      </td>
    </tr>
  </tbody>
</table>


<a id="preparation"></a>

## 🛠 Preparation

### Step 1: Clone the Repository

```bash
git clone https://github.com/Tencent-Hunyuan/OmniWeaving
cd OmniWeaving
```

### Step 2: Install Dependencies

OmniWeaving is built upon [HunyuanVideo-1.5](https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5). The way to install dependencies is similar to HunyuanVideo-1.5. Specifically, you should install basic dependencies:

```bash
pip install -r requirements.txt
```

Additionally, install the attention libraries as needed (we use Flash Attention in practice):

* **Flash Attention**: Install for faster inference and reduced GPU memory consumption. See [Flash Attention](https://github.com/Dao-AILab/flash-attention) for details.

* **Flex-Block-Attention**: Required only for sparse attention to achieve faster inference:
  ```bash
  git clone https://github.com/Tencent-Hunyuan/flex-block-attn.git
  cd flex-block-attn
  git submodule update --init --recursive
  python3 setup.py install
  ```

* **SageAttention**: For faster inference (will automatically disable Flex-Block-Attention):
  ```bash
  git clone https://github.com/cooper1637/SageAttention.git
  cd SageAttention 
  export EXT_PARALLEL=4 NVCC_APPEND_FLAGS="--threads 8" MAX_JOBS=32 # Optional
  python3 setup.py install
  ```

### Step 3: Download Models

**OmniWeaving**: Download from [HuggingFace](https://huggingface.co/tencent/HY-OmniWeaving).

Detailed download instructions are available at [download-checkpoint.md](download-checkpoint.md).


<a id="inference"></a>

## 🔑 Inference

In our inference code, we define **six task flags** corresponding to the [Supported Tasks](#supported-tasks). Their mapping is as follows:

| Task Flag | Full Name | Description |
|-----------|-----------|-------------|
| `t2v` | Text-to-Video | Generate videos from text prompts. |
| `i2v` | First-Frame-to-Video | Animate a static image into a video guided by text. |
| `interpolation` | Key-Frames-to-Video | Generate a video conditioned on start and end frames. |
| `reference2v` | Reference-to-Video / Compositional Multi-Image-to-Video | Single- or multi-subject reference-driven video generation. |
| `editing` | Video-to-Video Editing | Instruction-based video manipulation and stylization. |
| `tiv2v` | Text-Image-Video-to-Video | Generate a video conditioned on  text, image, and video inputs. |

Among these, **`t2v`, `i2v`, and `interpolation`** can optionally enable **thinking mode** (`--think`) for **Reasoning-Augmented Video Generation**, where the MLLM first reasons over user intent before generating the video.

### Common Configuration

All tasks share the following hyperparameters (configured at the top of `generate.sh`):

```bash
N_INFERENCE_GPU=8
SEED=0
ASPECT_RATIO=16:9
MODEL_PATH=/path/to/OmniWeaving

SAGE_ATTN=false ### Use Flash Attention
### SAGE_ATTN=true ### Use SageAttention
SPARSE_ATTN=false
OVERLAP_GROUP_OFFLOADING=false
ENABLE_CACHE=false
CACHE_TYPE=deepcache
```

> **Tips:** If your GPU memory is limited and you encounter OOM errors, try:
> ```bash
> export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128
> ```
> If you have limited CPU memory, disable overlapped group offloading by setting `OVERLAP_GROUP_OFFLOADING=false`.

### Task-Specific Inference Scripts

#### 1. Text-to-Video (`t2v`)

Generate a video from a text prompt.

```bash
PROMPT="Put Your Prompt Here"
NEGATIVE_PROMPT="overexposed, low quality, deformation, a poor composition, bad hands, bad teeth, bad eyes, bad limbs, distortion"
OUTPUT_PATH=./outputs/t2v.mp4

torchrun --nproc_per_node=$N_INFERENCE_GPU generate.py \
  --task t2v \
  --prompt "$PROMPT" \
  --negative_prompt "$NEGATIVE_PROMPT" \
  --aspect_ratio $ASPECT_RATIO \
  --seed $SEED \
  --sparse_attn $SPARSE_ATTN --use_sageattn $SAGE_ATTN \
  --enable_cache $ENABLE_CACHE --cache_type $CACHE_TYPE \
  --overlap_group_offloading $OVERLAP_GROUP_OFFLOADING \
  --output_path $OUTPUT_PATH \
  --model_path $MODEL_PATH \
  # --think \          # Optional: enable reasoning-augmented generation (see note below)
```

> The `--think` flag activates the MLLM's thinking mode, in which it reasons over user intent and generates an enriched prompt before video generation. The `--think` flag is supported by `t2v`, `i2v`, and `interpolation` tasks.

#### 2. First-Frame-to-Video (`i2v`)

Animate a first-frame image into a video guided by a text prompt.

```bash
PROMPT="Put Your Prompt Here"
IMAGE_PATH=/path/to/reference.png
OUTPUT_PATH=./outputs/i2v.mp4

torchrun --nproc_per_node=$N_INFERENCE_GPU generate.py \
  --task i2v \
  --prompt "$PROMPT" \
  --image_path $IMAGE_PATH \
  --aspect_ratio $ASPECT_RATIO \
  --seed $SEED \
  --sparse_attn $SPARSE_ATTN --use_sageattn $SAGE_ATTN \
  --enable_cache $ENABLE_CACHE --cache_type $CACHE_TYPE \
  --overlap_group_offloading $OVERLAP_GROUP_OFFLOADING \
  --output_path $OUTPUT_PATH \
  --model_path $MODEL_PATH \
  # --think \          # Optional: enable reasoning-augmented generation (see note below)
```

#### 3. Key-Frames-to-Video (`interpolation`)

Generate a video that bridges two key frames, guided by a text prompt.

```bash
PROMPT="Put Your Prompt Here"
REF_IMAGE_PATHS=(/path/to/first_frame.png /path/to/last_frame.png)
OUTPUT_PATH=./outputs/interpolation.mp4

torchrun --nproc_per_node=$N_INFERENCE_GPU generate.py \
  --task interpolation \
  --prompt "$PROMPT" \
  --ref_image_paths "${REF_IMAGE_PATHS[@]}" \
  --aspect_ratio $ASPECT_RATIO \
  --seed $SEED \
  --sparse_attn $SPARSE_ATTN --use_sageattn $SAGE_ATTN \
  --enable_cache $ENABLE_CACHE --cache_type $CACHE_TYPE \
  --overlap_group_offloading $OVERLAP_GROUP_OFFLOADING \
  --output_path $OUTPUT_PATH \
  --model_path $MODEL_PATH \
  # --think \          # Optional: enable reasoning-augmented generation
```

#### 4. Reference-to-Video / Compositional Multi-Image-to-Video (`reference2v`)

Generate a video featuring one or more reference subjects. Provide one or more reference images via `--ref_image_paths`.

```bash
PROMPT="Put Your Prompt Here"
# Supports 1–4 reference images.
# For best results with multiple images, use the same aspect ratio across all images,
# as they will be center-cropped to match the size of the first image.
REF_IMAGE_PATHS=(/path/to/img1.png /path/to/img2.png ... /path/to/img4.png)  # up to 4 input images
OUTPUT_PATH=./outputs/reference2v.mp4

torchrun --nproc_per_node=$N_INFERENCE_GPU generate.py \
  --task reference2v \
  --prompt "$PROMPT" \
  --ref_image_paths "${REF_IMAGE_PATHS[@]}" \
  --aspect_ratio $ASPECT_RATIO \
  --seed $SEED \
  --sparse_attn $SPARSE_ATTN --use_sageattn $SAGE_ATTN \
  --enable_cache $ENABLE_CACHE --cache_type $CACHE_TYPE \
  --overlap_group_offloading $OVERLAP_GROUP_OFFLOADING \
  --output_path $OUTPUT_PATH \
  --model_path $MODEL_PATH
```

#### 5. Video-to-Video Editing (`editing`)

Edit an existing video according to the text instruction (e.g., style transfer, object replacement).

```bash
PROMPT="Put Your Prompt Here"
CONDITION_VIDEO_PATH=/path/to/source_video.mp4
OUTPUT_PATH=./outputs/editing.mp4

# If you have pre-extracted VAE latents for the condition video, pass them via
# --condition_video_latents_path /path/to/latents.pt to skip VAE encoding at inference.
torchrun --nproc_per_node=$N_INFERENCE_GPU generate.py \
  --task editing \
  --prompt "$PROMPT" \
  --condition_video_paths $CONDITION_VIDEO_PATH \
  --aspect_ratio $ASPECT_RATIO \
  --seed $SEED \
  --sparse_attn $SPARSE_ATTN --use_sageattn $SAGE_ATTN \
  --enable_cache $ENABLE_CACHE --cache_type $CACHE_TYPE \
  --overlap_group_offloading $OVERLAP_GROUP_OFFLOADING \
  --output_path $OUTPUT_PATH \
  --model_path $MODEL_PATH \
  # --condition_video_latents_path /path/to/latents.pt  # Optional: skip VAE encoding by providing pre-extracted latents
```

#### 6. Text-Image-Video-to-Video (`tiv2v`)

Edit a video while incorporating reference subject images (e.g., insert a character from a reference image into a source video).

```bash
PROMPT="Put Your Prompt Here"
CONDITION_VIDEO_PATH=/path/to/source_video.mp4
# Only one reference image is supported for tiv2v.
# For best results, use a reference image whose aspect ratio is close to the output video's aspect ratio.
REF_IMAGE_PATHS=(/path/to/ref_image.png)
OUTPUT_PATH=./outputs/tiv2v.mp4

# If you have pre-extracted VAE latents for the condition video, pass them via
# --condition_video_latents_path /path/to/latents.pt to skip VAE encoding at inference.
torchrun --nproc_per_node=$N_INFERENCE_GPU generate.py \
  --task tiv2v \
  --prompt "$PROMPT" \
  --condition_video_paths $CONDITION_VIDEO_PATH \
  --ref_image_paths "${REF_IMAGE_PATHS[@]}" \
  --aspect_ratio $ASPECT_RATIO \
  --seed $SEED \
  --sparse_attn $SPARSE_ATTN --use_sageattn $SAGE_ATTN \
  --enable_cache $ENABLE_CACHE --cache_type $CACHE_TYPE \
  --overlap_group_offloading $OVERLAP_GROUP_OFFLOADING \
  --output_path $OUTPUT_PATH \
  --model_path $MODEL_PATH \
  # --condition_video_latents_path /path/to/latents.pt  # Optional: skip VAE encoding by providing pre-extracted latents
```

### Other Optional Arguments

The arguments below can be appended to any of the task commands above for further customization:

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--negative_prompt` | str | `""` | Negative prompt for video generation. Default is empty. Setting a negative prompt (e.g., `'overexposed, low quality, deformation, a poor composition, bad hands, bad teeth, bad eyes, bad limbs, distortion'`) can improve quality, especially for tasks like `t2v`. |
| `--num_inference_steps` | int | `50` | Number of denoising steps |
| `--video_length` | int | `81` | Number of frames to generate |
| `--fps` | int | Auto | Output FPS (default: 16 for ≤81 frames, 24 for >81 frames) |
| `--dtype` | str | `bf16` | Data type: `bf16` or `fp32` |
| `--offloading` | bool | `true` | Enable CPU offloading |
| `--group_offloading` | bool | `None` | Enable group offloading (auto-enabled with offloading) |
| `--pipeline_config` | str | `omniweaving` | Pipeline configuration preset that controls `guidance_scale` and `flow_shift`. Available presets: `omniweaving` (`guidance_scale=6.0, flow_shift=7.0`), `omniweaving2` (`guidance_scale=6.0, flow_shift=5.0`). |

> **Tuning `guidance_scale` / `flow_shift`:** You can switch presets via `--pipeline_config` (e.g., `--pipeline_config omniweaving2`). If the available presets do not meet your needs, you can add a new key to the [`PIPELINE_CONFIGS` dict in `hyvideo/commons/__init__.py`](hyvideo/commons/__init__.py#L48-L59) with your desired values. We recommend `guidance_scale=6.0` with `flow_shift=5.0` or `7.0`.


<a id="training-data-construction"></a>

## 🗂 Training Data Construction

To train OmniWeaving, beyond Foundational Video Generation Tasks, we design **Multimodal Composition Tasks** and **Reasoning-Augmented Tasks**. We provide reference data construction pipelines under [`process_data/`](process_data/) for three representative tasks: [Intent-Driven Image-to-Video](process_data/intent_driven_image_to_video/), [Text-Image-Video-to-Video](process_data/text_image_video_to_video/), and [Interleaved Text-and-Multi-Image-to-Video](process_data/interleaved_text_and_multi_image_to_video/). Each pipeline uses a VLM (e.g., Qwen3-VL-235B) for annotation, quality filtering, and instruction rewriting, with the final output saved in **Apache Arrow** format.

### Prerequisites

Launch a vLLM server for VLM inference (used by all pipelines). We use [Qwen3-VL-235B-A22B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-235B-A22B-Instruct) as the default VLM. Please refer to the [Qwen3-VL vLLM documentation](https://github.com/QwenLM/Qwen3-VL#vllm) for detailed deployment instructions. Below is an example:

```bash
vllm serve /path/to/Qwen3-VL-235B-A22B-Instruct \
  --tensor-parallel-size 8 \
  --enable-expert-parallel \
  --async-scheduling \
  --port 8080 \
  --max-model-len 65536 \
  --mm-encoder-tp-mode data \
  --dtype bfloat16 \
  --max-num-batched-tokens 256 \
  --distributed-executor-backend mp \
  --gpu-memory-utilization 0.9 \
  --limit-mm-per-prompt.video 2
```

### 1. Intent-Driven Image-to-Video

📂 [`process_data/intent_driven_image_to_video/`](process_data/intent_driven_image_to_video/)

A **Reasoning-Augmented Task**, where the model learns to formulate a reasoning trace detailing the temporal progression when visual and textual inputs lack explicit linkage in i2v tasks. This pipeline constructs training pairs of **(intent prompt, reasoning trace)** from raw videos. The pipeline for training data construction consists of **4 steps**:

| Step | Script | Description |
|------|--------|-------------|
| 1 | `step1.py` | Extract first frame from video; use VLM to generate motion descriptions. |
| 2 | `step2.py` | Use VLM to correct motion descriptions and predict user intent. |
| 3 | `step3.py` | VLM self-check: verify corrected motion and intent against the original video. |
| 4 | `step4_make_arrow.py` | Export valid training data to Arrow format. |

### 2. Text-Image-Video-to-Video

📂 [`process_data/text_image_video_to_video/`](process_data/text_image_video_to_video/)

A **Multimodal Composition Task**, where the inputs consist of three modalities (image, text, and video), requiring the model to seamlessly integrate target visual elements extracted from reference images into the temporal dynamics of a source video. This pipeline builds training data for edit-conditioned video generation from pairs of condition/edited videos. The pipeline for training data construction consists of **6 steps**:

| Step | Script | Description |
|------|--------|-------------|
| 1 | `step1.py` | Use VLM to judge whether the editing data is usable and extract a new-element description from before/after key frames of the video. |
| 2 | `step2.py` | Generate standalone subject images from edited frames using Flux. |
| 3 | `step3.py` | Use VLM to verify the quality of extracted reference images against the edited video. |
| 4 | `step4.py` | Use VLM to rewrite the editing instruction to concisely reference the reference image. |
| 5 | `step5.py` | Use VLM for final quality check: ensure consistency among reference image, rewritten instruction, and edited video. |
| 6 | `step6_make_arrow.py` | Export valid training data to Arrow format. |

### 3. Interleaved Text-and-Multi-Image-to-Video

📂 [`process_data/interleaved_text_and_multi_image_to_video/`](process_data/interleaved_text_and_multi_image_to_video/)

A **Multimodal Composition Task**, where the inputs contain multiple reference images (capturing key visual elements such as subjects or scenes) interleaved with text, requiring the model to accurately compose these elements into a cohesive video sequence. This pipeline decomposes videos into multi-subject compositions by extracting individual subject images and background. For simplicity, the provided pipeline currently omits the use of SAM3; please refer to the in-file comments for details. The pipeline for training data construction consists of **9 steps**:

| Step | Script | Description |
|------|--------|-------------|
| 1 | `step1.py` | Use VLM to identify main moving objects (up to 3) from video key frames and rewrite the instruction around them. |
| 2 | `step2.py` | Use VLM to rewrite the instruction by replacing object references with `[object1]`–`[object3]` placeholders. |
| 3 | `step3.py` | Use Flux to extract each identified object from the frame as standalone reference images. |
| 4 | `step4.py` | Use Flux to generate a background-only image with subjects removed. |
| 5 | `step5.py` | Use VLM to quality-check extracted reference images against the original frame. |
| 6 | `step6.py` | Use VLM to quality-check the background image (subjects properly removed, scene consistent). |
| 7 | `step7.py` | Merge results from previous steps, select the best reference image per object, and use VLM to produce the final instruction. |
| 8 | `step8.py` | Use VLM for final validation: verify that reference images and background align with the original video. |
| 9 | `step9_make_arrow.py` | Export valid training data to Arrow format. |


<a id="evaluation-on-intelligentvbench"></a>

## 📊 Evaluation on IntelligentVBench

We introduce **IntelligentVBench**, the first comprehensive benchmark for evaluating intelligent unified video generation. The benchmark covers four task categories with test cases provided under [`IntelligentVBench/csvs/`](intelligentVBench/csvs). All evaluations use **Gemini 2.5 Pro** as the judge model, scoring each sample on three dimensions (1–5 scale) and reporting both **AVG** (mean) and **MIN** (minimum) across dimensions.


### Benchmark Tasks

| Task | CSV | Description | Key Input Fields |
|------|-----|-------------|-----------------|
| **Implicit I2V** | `Implicit_I2V.csv` | First-frame-to-video with abstract/implicit text prompts | `ref_image_path`, `prompt` |
| **Interpolative DI2V** | `Interpolative_DI2V.csv` | Generate video bridging first and last frames | `first_frame`, `last_frame`, `prompt` |
| **TIV2V** | `TIV2V.csv` | Text-image-guided video editing | `condition_video`, `condition_image`, `prompt` |
| **Compositional MI2V** | `Compositional_MI2V_subject{1,2,3}.csv` | Multi-subject-and-scene compositional generation (1–3 subjects + background) | `subject_img_{1,2,3}`, `background_img`, `prompt` |

### CSV Field Descriptions

| Field | Used In | Description |
|-------|---------|-------------|
| `index` | All | Unique sample identifier; the generated video should be named `{index}.mp4` |
| `prompt` | All | Text instruction describing the desired video content or editing operation |
| `ref_image_path` | Implicit I2V | Path to the reference first-frame image to be animated |
| `first_frame` | Interpolative DI2V | Path to the starting frame image |
| `last_frame` | Interpolative DI2V | Path to the ending frame image |
| `edited_type` | TIV2V | Editing category: `local_change` (object replacement), `local_add` (add object), or `back_change` (background swap) |
| `condition_video` | TIV2V | Path to the source video to be edited |
| `condition_image` | TIV2V | Path to the reference image providing the target appearance for editing |
| `subject_img_1` | Compositional MI2V | Path to the first subject reference image |
| `subject_img_2` | Compositional MI2V (subject2, subject3) | Path to the second subject reference image |
| `subject_img_3` | Compositional MI2V (subject3) | Path to the third subject reference image |
| `background_img` | Compositional MI2V | Path to an optional background image (empty if not used) |

### Running Evaluation

**Step 1:** Download the IntelligentVBench test data from HuggingFace:

```bash
huggingface-cli download --repo-type dataset --resume-download midbee/IntelligentVBench \
  --local-dir ./IntelligentVBench \
  --local-dir-use-symlinks False
```

The downloaded dataset has the following structure:

```
IntelligentVBench/
├── csvs/                                   # Test case definitions
│   ├── Implicit_I2V.csv                    # 250 test cases
│   ├── Interpolative_DI2V.csv              # 250 test cases
│   ├── TIV2V.csv                           # 210 test cases
│   ├── Compositional_MI2V_subject1.csv     # 130 test cases (1 subject + background)
│   ├── Compositional_MI2V_subject2.csv     # 120 test cases (2 subjects + background)
│   └── Compositional_MI2V_subject3.csv     #  70 test cases (3 subjects + background)
├── images/
│   ├── Implicit_I2V/                       # Reference first-frame images ({index}.png)
│   ├── Interpolative_DI2V/                 # Start/end frame pairs ({index}_first.png, {index}_last.png)
│   ├── TIV2V/                              # Reference images for editing ({index}.png)
│   └── Compositional_MI2V/
│       ├── subject1/                       # Single subject images ({index}_subject1.png, optional {index}_background.png)
│       ├── subject2/                       # Two subject images ({index}_subject{1,2}.png, optional background)
│       └── subject3/                       # Three subject images ({index}_subject{1,2,3}.png, optional background)
└── videos/
    └── TIV2V/                              # Source videos for editing ({index}.mp4)
```

**Step 2:** Generate videos for each test case and save as `{index}.mp4` under an output directory.

> **Note:** Reference video generation scripts will be released soon!

**Step 3:** Set your Gemini API credentials in each evaluation script (`evaluate_implicit_i2v.py`, `evaluate_interpolative_di2v.py`, `evaluate_tiv2v.py`, `evaluate_compositional_mi2v.py`). Open the file and fill in the `api_key` and `gemini_url` fields at the top:

```python
api_key = "your-api-key"
gemini_url = "https://your-gemini-endpoint"
```

**Step 4:** Run the corresponding evaluation script:

```bash
# Implicit I2V
python IntelligentVBench/evaluate_implicit_i2v.py \
  --input_csv IntelligentVBench/csvs/Implicit_I2V.csv \
  --edit_path /path/to/generated_videos \
  --output_csv /path/to/implicit_i2v_eval.csv

# Interpolative DI2V
python IntelligentVBench/evaluate_interpolative_di2v.py \
  --input_csv IntelligentVBench/csvs/Interpolative_DI2V.csv \
  --edit_path /path/to/generated_videos \
  --output_csv /path/to/interpolative_di2v_eval.csv

# TIV2V
python IntelligentVBench/evaluate_tiv2v.py \
  --input_csv IntelligentVBench/csvs/TIV2V.csv \
  --edit_path /path/to/generated_videos \
  --output_csv /path/to/tiv2v_eval.csv

# Compositional MI2V
python IntelligentVBench/evaluate_compositional_mi2v.py \
  --input_csv1 IntelligentVBench/csvs/Compositional_MI2V_subject1.csv \
  --input_csv2 IntelligentVBench/csvs/Compositional_MI2V_subject2.csv \
  --input_csv3 IntelligentVBench/csvs/Compositional_MI2V_subject3.csv \
  --edit_path1 /path/to/subject1_videos \
  --edit_path2 /path/to/subject2_videos \
  --edit_path3 /path/to/subject3_videos \
  --output_dir /path/to/compositional_mi2v_eval
```

> **`--file_parent_path` option:** The image and video paths stored in the CSV files (e.g., `ref_image_path`, `first_frame`, `condition_video`, `subject_img_1`, etc.) are relative paths. You can use `--file_parent_path` to prepend a root to all relative paths. For example:
> ```bash
> python IntelligentVBench/evaluate_implicit_i2v.py \
>   --input_csv IntelligentVBench/csvs/Implicit_I2V.csv \
>   --edit_path /path/to/generated_videos \
>   --output_csv /path/to/implicit_i2v_eval.csv \
>   --file_parent_path /path/to/dataset/root
> ```
> If `--file_parent_path` is not specified (default `None`), the paths in the CSV are used as-is.

For Compositional MI2V, the `--output_dir` will produce the following structure:

```
/path/to/compositional_mi2v_eval/
├── subject1/
│   └── gemini-2.5-pro_Compositional_MI2V_subject1.csv
├── subject2/
│   └── gemini-2.5-pro_Compositional_MI2V_subject2.csv
└── subject3/
    └── gemini-2.5-pro_Compositional_MI2V_subject3.csv
```

**Step 5:** Calculate aggregated scores:

```bash
# For Implicit I2V (change --input_csv, --output_csv, --task_type accordingly for other tasks)
python IntelligentVBench/calculate_score.py \
  --input_csv IntelligentVBench/csvs/Implicit_I2V.csv \
  --output_csv /path/to/implicit_i2v_eval.csv \
  --task_type implicit_i2v  # options: implicit_i2v, interpolative_di2v, tiv2v

# For Compositional MI2V
python IntelligentVBench/calculate_score_compositional_mi2v.py \
  --input_csv1 IntelligentVBench/csvs/Compositional_MI2V_subject1.csv \
  --input_csv2 IntelligentVBench/csvs/Compositional_MI2V_subject2.csv \
  --input_csv3 IntelligentVBench/csvs/Compositional_MI2V_subject3.csv \
  --output_csv1 /path/to/compositional_mi2v_eval/subject1/gemini-2.5-pro_Compositional_MI2V_subject1.csv \
  --output_csv2 /path/to/compositional_mi2v_eval/subject2/gemini-2.5-pro_Compositional_MI2V_subject2.csv \
  --output_csv3 /path/to/compositional_mi2v_eval/subject3/gemini-2.5-pro_Compositional_MI2V_subject3.csv
```

<a id="examples"></a>

## 🎬 Examples

> For more qualitative examples across all tasks, please refer to our [project page](https://omniweaving.github.io/).

<table>
  <thead>
    <tr>
      <th>Task</th>
      <th>Input</th>
      <th>Output</th>
    </tr>
  </thead>
  <tbody>
    <!-- T2V -->
    <tr>
      <td rowspan="2"><b>Text-to-Video (T2V)</b></td>
      <td align="center"><details><summary>📋 Show prompt</summary>A beautiful Chinese woman with long hair, wearing sunglasses and a long fur coat over a fitted knit dress and high heels, slowly lay down in the thick snow.</details></td>
      <td align="center"><img src="assets/cases/t2v/2.gif" width="200"/></td>
    </tr>
    <tr>
      <td align="center"><details><summary>📋 Show prompt</summary>In the tranquil waters of the lake, a crocodile stealthily lurks, gliding through the depths. Gradually, it emerges, its massive, scaled head breaking the surface.</details></td>
      <td align="center"><img src="assets/cases/t2v/3.gif" width="200"/></td>
    </tr>
    <!-- I2V -->
    <tr>
      <td rowspan="2"><b>First-Frame-to-Video (I2V)</b></td>
      <td align="center"><img src="assets/cases/i2v/1.png" width="150"/><br/><details><summary>📋 Show prompt</summary>The woman is pleading for her life while being threatened with a firearm.</details></td>
      <td align="center"><img src="assets/cases/i2v/1.gif" width="200"/></td>
    </tr>
    <tr>
      <td align="center"><img src="assets/cases/i2v/2.png" width="150"/><br/><details><summary>📋 Show prompt</summary>Highlight the facial expressions and reactions of women after experiencing terrible events.</details></td>
      <td align="center"><img src="assets/cases/i2v/2.gif" width="200"/></td>
    </tr>
    <!-- Key-Frames-to-Video -->
    <tr>
      <td rowspan="2"><b>Key-Frames-to-Video</b></td>
      <td align="center"><img src="assets/cases/interpolation/1_first.png" width="70"/> <img src="assets/cases/interpolation/1_last.png" width="70"/><br/><details><summary>📋 Show prompt</summary>Camera tilts up to reveal giant anime girl's face.</details></td>
      <td align="center"><img src="assets/cases/interpolation/1.gif" width="200"/></td>
    </tr>
    <tr>
      <td align="center"><img src="assets/cases/interpolation/2_first.png" width="70"/> <img src="assets/cases/interpolation/2_last.png" width="70"/><br/><details><summary>📋 Show prompt</summary>Man turns head toward camera, smiling subtly with mouth slightly open.</details></td>
      <td align="center"><img src="assets/cases/interpolation/2.gif" width="200"/></td>
    </tr>
    <!-- V2V Editing -->
    <tr>
      <td rowspan="2"><b>Video-to-Video Editing</b></td>
      <td align="center"><img src="assets/cases/editing/1_before.gif" width="150"/><br/><details><summary>📋 Show prompt</summary>Convert the sketch-animated video to a live-action, photorealistic format.</details></td>
      <td align="center"><img src="assets/cases/editing/1_after.gif" width="200"/></td>
    </tr>
    <tr>
      <td align="center"><img src="assets/cases/editing/2_before.gif" width="150"/><br/><details><summary>📋 Show prompt</summary>Change man's hair color to platinum blonde while keeping hair shape.</details></td>
      <td align="center"><img src="assets/cases/editing/2_after.gif" width="200"/></td>
    </tr>
    <!-- Reference-to-Video -->
    <tr>
      <td><b>Reference-to-Video</b></td>
      <td align="center"><img src="assets/cases/reference2v/1.png" width="100"/><br/><details><summary>📋 Show prompt</summary>The man in the image, initially speaks with a sad expression. As he continues, he gently tilts his head down toward the other person.</details></td>
      <td align="center"><img src="assets/cases/reference2v/1.gif" width="200"/></td>
    </tr>
    <!-- Multi-Image-to-Video -->
    <tr>
      <td rowspan="2"><b>Compositional Multi-Image-to-Video</b></td>
      <td align="center"><img src="assets/cases/compositional/1_1.png" width="45"/> <img src="assets/cases/compositional/1_2.png" width="45"/> <img src="assets/cases/compositional/1_3.png" width="45"/><br/><details><summary>📋 Show prompt</summary>The woman in the first image holds the handbag in the second image up towards the camera while speaking over the background in the third image.</details></td>
      <td align="center"><img src="assets/cases/compositional/1.gif" width="200"/></td>
    </tr>
    <tr>
      <td align="center"><img src="assets/cases/compositional/2_1.png" width="35"/> <img src="assets/cases/compositional/2_2.png" width="35"/> <img src="assets/cases/compositional/2_3.png" width="35"/> <img src="assets/cases/compositional/2_4.png" width="35"/><br/><details><summary>📋 Show prompt</summary>The pink handbag in the first image, the green handbag in the second image, and the blue handbag in the third image remain stationary but shift slightly in the frame due to the camera's slow upward movement over the background in the fourth image.</details></td>
      <td align="center"><img src="assets/cases/compositional/2.gif" width="200"/></td>
    </tr>
    <!-- TIV2V -->
    <tr>
      <td rowspan="2"><b>Text-Image-Video-to-Video</b></td>
      <td align="center"><img src="assets/cases/tiv2v/1_ref.png" width="50"/><br/><img src="assets/cases/tiv2v/1_before.gif" width="150"/><br/><details><summary>📋 Show prompt</summary>Replace the man in the video with the man in the image.</details></td>
      <td align="center"><img src="assets/cases/tiv2v/1_after.gif" width="200"/></td>
    </tr>
    <tr>
      <td align="center"><img src="assets/cases/tiv2v/2_ref.png" width="50"/><br/><img src="assets/cases/tiv2v/2_before.gif" width="150"/><br/><details><summary>📋 Show prompt</summary>Add the coffee mug in the image onto the desk near the keyboard in the video.</details></td>
      <td align="center"><img src="assets/cases/tiv2v/2_after.gif" width="200"/></td>
    </tr>
    <!-- Reasoning -->
    
  </tbody>
</table>


<a id="citation"></a>

## 📚 Citation

If you find our work helpful, please consider giving us a star ⭐ on this repo and citing our papers as follows:

**OmniWeaving**

```bibtex
@article{pan2026omniweaving,
  title={OmniWeaving: Towards Unified Video Generation with Free-form Composition and Reasoning},
  author={Pan, Kaihang and Tian, Qi and Zhang, Jianwei and Kong, Weijie and Xiong, Jiangfeng and Long, Yanxin and Zhang, Shixue and Qiu, Haiyi and Wang, Tan and Lv, Zheqi and others},
  journal={arXiv preprint arXiv:2603.24458},
  year={2026}
}
```

**HunyuanVideo 1.5**

```bibtex
@article{wu2025hunyuanvideo,
  title={Hunyuanvideo 1.5 technical report},
  author={Wu, Bing and Zou, Chang and Li, Changlin and Huang, Duojun and Yang, Fang and Tan, Hao and Peng, Jack and Wu, Jianbing and Xiong, Jiangfeng and Jiang, Jie and others},
  journal={arXiv preprint arXiv:2511.18870},
  year={2025}
}
```

<a id="acknowledgements"></a>

## 🙏 Acknowledgements
We would like to thank the contributors to [HunyuanVideo 1.5](https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5), [Transformers](https://github.com/huggingface/transformers), [Diffusers](https://github.com/huggingface/diffusers), [HuggingFace](https://huggingface.co/) and [Qwen-VL](https://github.com/QwenLM/Qwen-VL), for their open research and exploration.

