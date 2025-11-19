# Moodifier: MLLM-Enhanced Emotion-Driven Image Editing

<p align="center">
  <!-- Paper (generic, non-arXiv) -->
  <a href="https://moodifymturkbucket.s3.us-east-2.amazonaws.com/IEEE_Transactions_on_Affective_Computing+(1).pdf">
    <img src="https://img.shields.io/badge/Paper-PDF-critical" alt="Paper">
  </a>
  <!-- Project page -->
  <a href="https://emomoodify.github.io/app/">
    <img src="https://img.shields.io/badge/Project-Page-orange" alt="Project Page">
  </a>
  <!-- Hugging Face model -->
  <a href="https://huggingface.co/emomoodify/moodifyCLIP/tree/master">
    <img src="https://img.shields.io/badge/HuggingFace-Model-yellow" alt="Hugging Face Model">
  </a>
  <!-- Dataset / MoodArchive -->
  <a href="https://drive.google.com/drive/folders/1hhy2p50FmdaoIAnFqnOeacZGWgHnZDb0?usp=sharing">
    <img src="https://img.shields.io/badge/Dataset-MoodArchive-brightgreen" alt="Dataset: MoodArchive">
  </a>
  <!-- License -->
  <a href="https://opensource.org/licenses/CC-BY-4.0">
    <img src="https://img.shields.io/badge/License-CC%20BY%204.0-blue.svg" alt="License">
  </a>
</p>

> **One image, many moods in one click.**  
> Moodifier keeps the objects and scene *fixed* and edits only the **emotional appearance** (colors, lighting, style, expressions, and local details) to match a target emotion.

https://github.com/user-attachments/assets/e5d7a0c4-3d3a-4c2e-b82e-722c9e0a87bb

---

## 🔍 What is Moodifier?

Moodifier is a **training-free emotion-driven image editor** that combines:

1. **MoodArchive** – a large-scale affective image dataset (8.4M+ images, 27 emotions × 4 visual contexts). 
2. **MoodifyCLIP** – a CLIP-style vision–language model fine-tuned on MoodArchive for nuanced emotional reasoning. 
3. **Moodifier Editor** – an MLLM-enhanced diffusion editing pipeline that performs **precise emotional transformations** while preserving identity, structure, and background. 

The system supports diverse use cases such as:

* Character expression editing (keep identity, change emotion)
* Fashion & jewelry design
* Product mockups & e-commerce imagery
* Home décor and mood exploration

**[Project page](https://emomoodify.github.io/app/index.html#blocks)** 

---

## 📂 Repository Structure

A typical layout (simplified):

```text
moodifier/
├── diffusers/               # Patched / custom diffusion pipelines & attention control
├── groundingdino/           # GroundingDINO utilities for object detection & masks
├── models/                  # Model configs and weight loading helpers
├── ptp_utils.py             # Prompt-to-Prompt / attention editing utilities
├── run_editing_demoapp.py   # Demo app entry point (Gradio / UI runner)
├── utils.py                 # Shared helpers (IO, visualization, masks, etc.)
├── LICENSE
└── README.md
```

> See the individual modules for implementation details and available options.

---

## ⚙️ Installation

### 1. Environment

Recommended:

* Python **3.9+**
* PyTorch **with CUDA** (2.0+ recommended)
* A GPU with ≥12GB VRAM for comfortable 512–768px editing

Typical Python dependencies (names may vary by release):

* `torch`, `torchvision`, `torchaudio`
* `diffusers`
* `transformers`
* `accelerate`
* `sentencepiece`
* `opencv-python`, `Pillow`
* `groundingdino` or equivalent GroundingDINO wrapper
* `gradio` (for the demo app)

### 2. Setup

```bash
git clone https://github.com/emomoodify/moodifier.git
cd moodifier

# Create a new environment (example with conda)
conda create -n moodifier python=3.10
conda activate moodifier

# Install the required packages
pip install -r requirements.txt   # if provided
# or install the core libraries manually as listed above
```

### 3. Model Weights

You will need access to:

* A Stable Diffusion checkpoint (e.g., `CompVis/stable-diffusion-v1-4`). 
* A LLaVA-NeXT or similar MLLM checkpoint. 
* MoodifyCLIP weights (ViT-L/14), if not bundled in this repo. 

Please follow the instructions in the `models/` directory or project page for download links and licensing details.

---

## 🚀 Quick Start

### 1. Launch the Demo App

Most users will start from the interactive demo:

```bash
python run_editing_demoapp.py
```

This typically:

1. Opens a local Gradio (or similar) interface.
2. Lets you:

   * Upload a **source image**
   * Choose a **target emotion** (e.g., `joy`, `sadness`, `excitement`)
   * Optionally refine regions to edit (e.g., face only, selected object)
3. Returns the **emotionally edited image**.

> For available command-line flags, check `python run_editing_demoapp.py --help` or the script header comments.

### 2. Programmatic Usage (Conceptual)

A typical programmatic flow inside your own script:

```python
from diffusers import StableDiffusionPipeline
from models.moodifyclip import MoodifyCLIP
from utils import load_mllm, run_moodifier_edit

# 1. Load models
sd_pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4").to("cuda")
moodifyclip = MoodifyCLIP.from_pretrained("path/to/moodifyclip")
mllm = load_mllm("llava-next-path")

# 2. Run a single edit
edited = run_moodifier_edit(
    image="input.jpg",
    target_emotion="relief",
    sd_pipe=sd_pipe,
    moodifyclip=moodifyclip,
    mllm=mllm,
)

edited.save("output_relief.jpg")
```

(Exact function names may differ; please refer to the actual code.)

---

## 📊 Evaluation (High-Level Summary)

Moodifier is evaluated along two axes: 

1. **Representation quality (MoodifyCLIP)**

   * Zero-shot emotion classification on Emotion6, EmoSet, Emotic
   * Image–text and text–image retrieval on SentiCap, Affection, and a human-verified MoodArchive-5k subset
2. **Editing quality (Moodifier)**

   * Structural preservation:

     * Structural distance, PSNR, SSIM, LPIPS, MSE
   * Emotional correctness:

     * CLIP similarity between edited images and emotion prompts
   * Human preference study (MTurk)
---

## 🧪 Research Use Cases

You can build on Moodifier for:

* Emotion-aware data augmentation for vision & affective computing.
* User studies on how emotional tones impact behavior (e.g., click-through rates in e-commerce).
* Emotion-conditioned generative design tools for fashion, jewelry, and interior design.
* Exploring emotional alignment of diffusion models and MLLMs.

---

## 📄 License

This repository is released under the license specified in [`LICENSE`](./LICENSE).
Please also respect the licenses of all upstream models and datasets (Stable Diffusion, LLaVA-NeXT, CLIP, GroundingDINO, image sources used for MoodArchive, etc.).

---

## 🙏 Acknowledgements

Moodifier builds upon the excellent work of many open-source projects and datasets, including but not limited to:

* **Stable Diffusion** (CompVis / Stability AI) 
* **CLIP** (OpenAI) and its CommonPool, DataComp, LAION variants 
* **LLaVA-NeXT** for multimodal captioning and reasoning 
* **GroundingDINO** for text-conditioned object detection
* Public image sources such as Unsplash, Pexels, Pixabay, Envato Elements, Openverse, Shopify Burst, Stocksnap, FoodiesFeed, and FreeNatureStock. 

---
