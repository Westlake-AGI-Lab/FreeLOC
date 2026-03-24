<!-- <p align="center">
    <img src="assets/logo.png" width="30%">
</p> -->

# <div align="center">FreeLOC: Free-Lunch Long Video Generation via Layer-Adaptive O.O.D Correction<br><sup>(CVPR 2026)</sup></div>

<div align="center">
  <p>
    <a href="">Jiahao Tian</a><sup>1</sup>&nbsp;&nbsp;
    <a href="">Chenxi Song</a><sup>1*</sup>&nbsp;&nbsp;
    <a href="">Wei Cheng</a><sup>1</sup>&nbsp;&nbsp;
    <a href="">Chi Zhang</a><sup>1†</sup>
  </p>
  <p>
    <sup>1</sup>AGI Lab, Westlake University&nbsp;&nbsp;
  </p>
  <p>
    <sup>*</sup>Project Leader&nbsp;&nbsp;
    <sup>†</sup>Corresponding Author
  </p>
</div>

<p align="center">
  <!-- <a href=''><img src='https://img.shields.io/badge/Project-Page-Green'></a> -->
  <!-- &nbsp; -->
  <a href=""><img src="https://img.shields.io/static/v1?label=Arxiv&message=FreeLOC&color=red&logo=arxiv"></a>
  &nbsp;
  <!-- <a href=''><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-orange'></a> -->
</p>

## 📰 News

- **[2025/03/18]** Release code for FreeLOC built on [Wan2.1-T2V-1.3B](https://github.com/Wan-Video/Wan2.1).

## 📷 Introduction

**TL;DR:**
We propose **FreeLOC**, a training-free, layer-adaptive framework that enables pre-trained short video diffusion transformers (DiTs) to generate high-fidelity long videos. FreeLOC identifies and addresses two critical out-of-distribution (O.O.D) problems that arise when extending video length beyond the training duration:

1. **Frame-level Relative Position O.O.D** — RoPE embeddings fall outside the training distribution, causing temporal inconsistency.
2. **Context-length O.O.D** — Extended token sequences dilute attention weights, degrading visual detail.

To tackle these, we introduce:
- **Video-based Relative Position Re-encoding (VRPR)**: a multi-granularity strategy that hierarchically remaps temporal relative positions back into the pre-trained range.
- **Tiered Sparse Attention (TSA)**: a hierarchical attention mechanism that preserves dense local attention while using progressively sparser patterns for distant frames.
- **Layer-Adaptive Probing**: an automatic mechanism that identifies each transformer layer's sensitivity to the two O.O.D sources, enabling selective and efficient application of VRPR and TSA.

<div align="center">
  <img src="assets/pipeline.png" width="95%">
</div>

## 🎬 Visual Comparison

<table>
<tr>
<th>Direct Sampling (4×)</th>
<th>FreeLOC (Ours, 4×)</th>
</tr>
<tr>
<td><video src="https://github.com/user-attachments/assets/eb64e8b3-e6f3-4be2-9838-454b92f93f6c" width="100%"></video></td>
<td><video src="https://github.com/user-attachments/assets/8db5017f-e774-414c-932c-958f3441570c
" width="100%"></video></td>
</tr>
<tr>
<td><video src="https://github.com/user-attachments/assets/2cb27e60-d442-47fb-8e86-5c6bed393510" width="100%"></video></td>
<td><video src="https://github.com/user-attachments/assets/6e20dae9-8bb8-4900-a745-76c7df51cedc" width="100%"></video></td>
</tr>
</table>


## 🛠️ Installation

**Requirements**
- NVIDIA GPU with > 28GB VRAM
- Linux operating system
- CUDA 12.4+, Python 3.10+

**Environment**

We provide a pre-configured conda environment. Alternatively, you can set up from scratch:

```bash
git clone https://github.com/xxx/FreeLOC.git
cd FreeLOC

# Option 2: Create from scratch
conda create -n freeloc python=3.10 -y
conda activate freeloc
pip install torch>=2.4.0 torchvision>=0.19.0
pip install -r requirements.txt
pip install flash-attn --no-build-isolation
```

## 🧱 Download Checkpoints

Download Wan2.1 model checkpoints:

```bash
pip install "huggingface_hub[cli]"

# Wan2.1-T2V-1.3B
huggingface-cli download Wan-AI/Wan2.1-T2V-1.3B --local-dir /path/to/Wan2.1-T2V-1.3B
```

Or using ModelScope mirror (for users in China):

```bash
huggingface-cli download Wan-AI/Wan2.1-T2V-1.3B --local-dir /path/to/Wan2.1-T2V-1.3B
```

## 🔑 Inference

**FreeLOC Long Video Generation**

```bash
python generate_freeloc.py \
  --task t2v-1.3B \
  --size 832*480 \
  --ckpt_dir /path/to/Wan2.1-T2V-1.3B \
  --offload_model True --t5_cpu \
  --sample_shift 8 \
  --sample_guide_scale 6 \
  --frame_num 161 \
  --base_seed 42 \
  --prompt "A focused man in a sleek, black athletic outfit sprints along a winding forest trail, surrounded by towering trees and dappled sunlight filtering through the leaves." \
  --runtime_config wan/configs/freeloc_config.json
```

FreeLOC's behavior is controlled via `--runtime_config`, which points to a JSON file (default: `wan/configs/freeloc_config.json`). The frame count `--frame_num` must satisfy 4n+1 (e.g., 81, 161, 321).

## ⚙️ Runtime Configuration

FreeLOC's behavior is controlled via a JSON runtime config (`wan/configs/freeloc_config.json`). Key settings:

```jsonc
{
  "text2video": {
    "model_forward_kwargs": {
      "rope_type": null,                 // RoPE variant; null = default
      "enable_rp_map": true,             // Enable VRPR relative position re-encoding
      "enable_layer_modify": true,       // Enable layer-adaptive O.O.D correction
      "relative_map_mod": "vrpr",        // Re-encoding mode ("vrpr")
      "use_radial_attention": true,      // Enable Tiered Sparse Attention (TSA)
      "rope_relative_layers": null       // Per-run layer override; null = use default_rope_relative_layers
    },
  },
  "model": {
    // DiT layers identified as sensitive to O.O.D via automatic probing
    "default_rope_relative_layers": [0, 1, 4, 6, ...],
    "radial_attention": {                // TSA parameters
      "window_size_1": 8,               // D1: local dense attention window (frames)
      "window_size_2": 24,              // D2: mid-range striped attention boundary (frames)
      "fallback_multiplier": 12,        // Multiplier for fallback window when TSA is disabled
      "rope_shift": {                   // VRPR parameters used inside TSA layers
        "group_size": 8,
        "window_size": 12            
      },
      "diag_size_token_per_frame": 1560 // Spatial tokens per frame (832×480 → 1560)
    },
    "rp_map": {                          // VRPR parameters
      "diag_size_token_per_frame": 1560, 
      "vrpr_clip_max": 20,              // Max relative position after re-encoding
      "vrpr": {
        "frame_settings": {              // Per-frame-count VRPR granularity settings
          "41": {                        // Settings when latent frame count = 41 
            "window_size_1": 12,         // W1: fine-grained window boundary
            "window_size_2": 20,         // W2: medium-grained window boundary
            "group_size_1": 2,           // G1: medium-grained quantization group size
            "group_size_2": 8            // G2: coarse-grained quantization group size
          },
          "81": {                        // Settings when latent frame count = 81
            "window_size_1": 10,
            "window_size_2": 14,
            "group_size_1": 2,
            "group_size_2": 8
          }
        }
      }
    }
  }
}
```

## 📊 Results

Quantitative comparison at 4× video length extension on Wan2.1-T2V-1.3B (321 frames, [VBench](https://github.com/Vchitect/VBench) metrics):

| Method | SC ↑ | BC ↑ | MS ↑ | IQ ↑ | AQ ↑ | DD ↑ |
|--------|------|------|------|------|------|------|
| Direct Sampling | 98.50 | 97.89 | 98.83 | 59.21 | 49.43 | 4.32 |
| Sliding Window | 96.15 | 95.92 | 98.54 | 65.64 | 54.04 | 39.81 |
| RIFLEx | 98.41 | 97.87 | 98.86 | 59.92 | 49.67 | 4.45 |
| FreeLong | 97.88 | 97.51 | 98.91 | 63.17 | 54.56 | 21.21 |
| FreeNoise | 97.31 | 97.25 | 98.84 | 66.32 | 56.01 | 35.11 |
| **FreeLOC (Ours)** | **98.44** | **97.78** | **98.97** | **67.44** | **61.21** | **36.27** |

> SC: Subject Consistency, BC: Background Consistency, MS: Motion Smoothness, IQ: Imaging Quality, AQ: Aesthetic Quality, DD: Dynamic Degree.


## 🤗 Acknowledgement

- [Wan2.1](https://github.com/Wan-Video/Wan2.1): the base video generation model we built upon.
- [VBench](https://github.com/Vchitect/VBench): the evaluation benchmark.

## 🌟 Citation

If you find this work useful, please cite our paper:

```bibtex
@article{tian2025freeloc,
  title={Free-Lunch Long Video Generation via Layer-Adaptive O.O.D Correction},
  author={Tian, Jiahao and Song, Chenxi and Cheng, Wei and Zhang, Chi},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2026}
}
```
