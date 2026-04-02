
# Download the pretrained checkpoints:

First, make sure you have installed the huggingface CLI and modelscope CLI.

```bash
pip install -U "huggingface_hub[cli]"
pip install modelscope
```


### Download OmniWeaving checkpoints:

> **Note:** OmniWeaving is trained based on HunyuanVideo 1.5. The released weight file structure is similar to HunyuanVideo 1.5, with an additional `text_encoder/ckpt` folder that stores the fine-tuned MLLM weights (based on Qwen2.5-VL-7B-Instruct).

```bash
hf download tencent/HY-OmniWeaving --local-dir /path/to/ckpts
```

### Downloading Text Encoders and Vision Encoders

OmniWeaving uses a fine-tuned MLLM and a byT5 as text encoders.

* **MLLM**

    OmniWeaving's MLLM uses Qwen2.5-VL-7B-Instruct as the base model. You need to download it by the following command:
    ```bash
    hf download Qwen/Qwen2.5-VL-7B-Instruct --local-dir /path/to/ckpts/text_encoder/llm
    ```

* **ByT5 encoder**

    OmniWeaving uses [Glyph-SDXL-v2](https://modelscope.cn/models/AI-ModelScope/Glyph-SDXL-v2) as our [byT5](https://github.com/google-research/byt5) encoder, which can be downloaded by the following command:

    ```bash
    hf download google/byt5-small --local-dir /path/to/ckpts/text_encoder/byt5-small
    modelscope download --model AI-ModelScope/Glyph-SDXL-v2 --local_dir /path/to/ckpts/text_encoder/Glyph-SDXL-v2
    ```

* **Vision Encoder**
    We use Siglip as the vision encoder. To download this model, you need to request access to the [FLUX.1-Redux-dev model](https://huggingface.co/black-forest-labs/FLUX.1-Redux-dev) on the Hugging Face website. After your request is approved, use your personal Hugging Face access token to download the model as follows (please replace `<your_hf_token>` with your actual Hugging Face access token):
    ```bash
    hf download black-forest-labs/FLUX.1-Redux-dev --local-dir /path/to/ckpts/vision_encoder/siglip --token <your_hf_token>
    ```
### Final Checkprint Structure

After downloading all the checkpoints, the final file structure should look like this:

```
/path/to/ckpts
├── text_encoder
│   ├── ckpt                        # Fine-tuned MLLM weights (from OmniWeaving)
│   │   └── text_encoder_model.safetensors
│   ├── llm                         # Qwen2.5-VL-7B-Instruct (base model)
│   │   └── ...
│   ├── byt5-small                  # byT5 encoder
│   │   └── ...
│   └── Glyph-SDXL-v2               # Glyph-SDXL-v2
│       └── assets
│  
├── vision_encoder
│   └── siglip                      # SigLIP vision encoder
│       └── ...
│ 
├── scheduler                       # from OmniWeaving
├── upsampler                       # from OmniWeaving
├── vae                             # from OmniWeaving
├── transformer                     # from OmniWeaving (containing model weights)
└── config.json                     # from OmniWeaving
```


<details>

<summary>💡Tips for using hf/huggingface-cli (network problem)</summary>

##### 1. Using HF-Mirror

If you encounter slow download speeds in China, you can try a mirror to speed up the download process:

```shell
HF_ENDPOINT=https://hf-mirror.com hf download tencent/HunyuanVideo-1.5 --local-dir /path/to/ckpts
```

##### 2. Resume Download

`huggingface-cli` supports resuming downloads. If the download is interrupted, you can just rerun the download 
command to resume the download process.

Note: If an `No such file or directory: 'ckpts/.huggingface/.gitignore.lock'` like error occurs during the download 
process, you can ignore the error and rerun the download command.

</details>