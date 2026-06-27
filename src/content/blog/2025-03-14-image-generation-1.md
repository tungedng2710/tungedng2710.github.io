---
title: "AI Art Tutorial: High quality image generation"
pubDate: 2025-03-14
image: "https://images.rawpixel.com/editor_1024/cHJpdmF0ZS9sci9pbWFnZXMvd2Vic2l0ZS8yMDIyLTA1L2ZyZmFudGFzeV9sYW5kc2NhcGVfZHJ5X21hcnMtaW1hZ2Uta3liZHNyY2suanBn.jpg"
description: ComfyUI is a powerful and modular GUI and backend for stable diffusion models, featuring a graph/node-based interface that allows you to design and execute advanced stable diffusion workflows without any coding
tags:
- Generative AI
- AI Art
- ComfyUI
authorName: Tung Nguyen
authorUrl: https://github.com/tungedng2710
---


# AI Art Tutorial: High quality image generation with ComfyUI

In the ever-evolving landscape of artificial intelligence, tools like ComfyUI are empowering artists and enthusiasts to create breathtaking, one-of-a-kind images. By leveraging cutting-edge models and a user-friendly interface, ComfyUI turns the once-complicated task of AI art generation into a streamlined, visually intuitive process. Whether you’re new to AI art or looking for a more efficient workflow, this tutorial will guide you step-by-step to produce high-quality, captivating images with ease. Join us as we explore the essential techniques, settings, and tips to unlock your creativity and make the most of ComfyUI.

### **Tech Stack & Requirements**  

To achieve high-quality image generation with **ComfyUI**, this tutorial leverages the following technologies:  

- **ComfyUI Framework** – A node-based AI image generation tool for flexibility and control.  
- **FLUX.1 Dev** – The AI model powering advanced image synthesis.  
- **T5XXL & CLIP Text Encoders** – Enhancing text-to-image generation with precise prompt understanding.  
- **Real-ESRGAN Models** – Used for upscaling images to 4K resolution while maintaining detail and sharpness.  

#### **Hardware Requirements**  
To ensure smooth performance, a **GPU with at least 24GB VRAM** is required. Recommended options:  
- **NVIDIA RTX 3090** (Minimum requirement) 
- **NVIDIA RTX 4090** (for better performance and faster processing)  

Having a **powerful GPU** is essential for handling large model weights and upscaling operations efficiently. Of course, if you’re feeling *extravagant* (or just casually swimming in cash), a **data center GPU like the B200 Blackwell** will supercharge your AI art like a jet engine on a tricycle—because why wait when you can render masterpieces at warp speed? 🚀🔥

## Step 1 – Install ComfyUI

ComfyUI is a desktop app, so you’ll run it locally on your machine. On most systems, the basic setup looks like this:

1. Install Python (3.10+ is recommended) and the latest GPU‑enabled PyTorch for your OS and CUDA version.  
2. Clone the ComfyUI repository:
   ```bash
   git clone https://github.com/comfyanonymous/ComfyUI.git
   cd ComfyUI
   ```
3. (Optional but recommended) Create and activate a virtual environment.  
4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
5. Start ComfyUI:
   ```bash
   python main.py
   ```

When the server starts, open your browser and navigate to `http://127.0.0.1:8188`. You should see the node‑based canvas where we’ll build our image generation pipeline.

> Note: Installation details can change over time. If you run into errors, check the ComfyUI GitHub README for up‑to‑date instructions for your OS.

## Step 2 – Download the models

To follow this tutorial, you’ll need four main pieces:

- **FLUX.1 Dev checkpoint** (the core generative model).  
- **T5XXL text encoder** (for understanding prompts).  
- **CLIP text encoder** (for additional conditioning / negative prompts).  
- **Real‑ESRGAN upscaler** (for turning your base image into a sharp 4K render).

The exact folders may vary depending on your ComfyUI setup, but a common layout is:

- `ComfyUI/models/checkpoints` – for the **FLUX.1 Dev** model file.  
- `ComfyUI/models/text_encoders` – for **T5XXL** and **CLIP** weights.  
- `ComfyUI/models/upscale_models` – for **Real‑ESRGAN** `.pth` or `.onnx` files.  

After downloading, restart ComfyUI and you should see these models appear in the corresponding node dropdowns.

## Step 3 – Build a basic FLUX.1 workflow

Now comes the fun part: building a simple, clean workflow for high‑quality images. On the ComfyUI canvas, create the following graph:

1. **Model loader**
   - Add a **Load Checkpoint** (or FLUX‑specific loader) node.  
   - Select your **FLUX.1 Dev** checkpoint.  
   - This node typically outputs the **UNet/transformer**, **VAE**, and **CLIP/T5** handles used by the rest of the graph.

2. **Text encoders**
   - Add a **T5 Text Encode** (or equivalent) node for the **main prompt**.  
   - Add a **CLIP Text Encode (Prompt)** node for the main prompt (optional but recommended).  
   - Add a **CLIP Text Encode (Negative Prompt)** node for things you *don’t* want (e.g. extra limbs, text artifacts).  

3. **Latent image**
   - Add an **Empty Latent Image** node.  
   - Set your base resolution, for example:
     - `Width: 1024`  
     - `Height: 1024`  
   - We’ll upscale to 4K later, so starting at 1024×1024 or 1216×832 is a good balance between quality and VRAM usage.

4. **Sampler (denoising)**
   - Add a **KSampler** (or equivalent sampler used by your FLUX nodes).  
   - Connect:
     - The **model** output from the checkpoint loader.  
     - The **positive / negative conditioning** from your text encoder nodes.  
     - The **latent image** from `Empty Latent Image`.  
   - Suggested starting parameters:
     - `Steps`: 24–32  
     - `CFG scale`: 4–6  
     - `Sampler`: `Euler`, `DPM++ 2M`, or the sampler recommended for your FLUX build.  
     - `Seed`: 0 or `-1` (for random).  

5. **Decode & save**
   - Add a **VAE Decode** node and connect it to the sampler’s latent output.  
   - Add a **Save Image** node and connect the decoded image.  

At this point you already have a complete text‑to‑image pipeline. Hit **Queue Prompt**, wait for the sampler to finish, and you’ll get your first FLUX.1 Dev image.

## Step 4 – Prompting for high‑quality images

Model architecture and resolution matter, but **prompts** are still your main steering wheel. A good high‑quality prompt typically includes:

- **Subject** – what you want to see.  
- **Style** – photo, illustration, anime, 3D render, etc.  
- **Lighting & mood** – soft light, cinematic, moody, golden hour.  
- **Composition** – close‑up, wide shot, centered, rule of thirds.  
- **Technical descriptors** – 8k, ultra‑detailed, volumetric lighting (don’t overdo these).

Example prompt for a portrait:

```text
hyperrealistic portrait of a young woman, soft natural lighting,
50mm lens, shallow depth of field, freckles, detailed skin texture,
subtle makeup, cinematic warm tones, shot on film
```

Example negative prompt:

```text
blurry, low resolution, distorted hands, extra limbs, text, watermark,
logo, frame, jpeg artifacts, oversharpened
```

Tips:

- If the image feels **over‑cooked** or artificial, lower **CFG** slightly (e.g. from 7 → 4.5).  
- If the model ignores your instructions, try raising **steps** or slightly increasing CFG.  
- For consistent characters, reuse the same **seed** and only tweak the prompt gradually.

## Step 5 – Upscale to 4K with Real‑ESRGAN

Once you’re happy with the base image, it’s time to turn it into a crisp 4K render.

1. **Add the upscaler**
   - Insert a **Real‑ESRGAN Upscale** (or similar) node.  
   - Connect the decoded image (`VAE Decode` output) to the upscaler’s input.  
   - Choose a suitable model (e.g. `RealESRGAN_x4plus` for general images, or anime‑specific variants for stylized art).

2. **Set the upscale factor**
   - If your base resolution is **1024×1024**, use `Scale: 4x` to reach roughly **4096×4096**.  
   - For rectangular images (e.g. 1216×832), the output will scale proportionally but still be in the ~4K range on the longest side.

3. **Save the final result**
   - Connect the Real‑ESRGAN output to a **Save Image** node.  
   - Run the workflow again (or place the upscaler in a separate branch so you don’t resample the whole image).  

If you hit VRAM limits during upscaling:

- Try **2x upscaling** first, then a second pass.  
- Close other GPU‑heavy apps (browsers, games, etc.).  
- Reduce **batch size** to 1 and avoid tiling unless necessary.

## Step 6 – Fine‑tuning quality and avoiding common issues

Even with strong models like FLUX.1 Dev, you may encounter common AI‑art problems. Here are practical tweaks:

- **Anatomy / hands look weird**  
  - Strengthen the negative prompt: `deformed hands, extra fingers, fused fingers`.  
  - Use more modest adjectives instead of “insanely detailed” everywhere.  

- **Composition feels random**  
  - Add camera hints: `wide shot`, `full body`, `close‑up`, `over‑the‑shoulder`.  
  - Use a lower resolution (e.g. 832×1216) then upscale; smaller latent spaces are often easier to control.  

- **Noisy or muddy details**  
  - Increase steps slightly (e.g. from 24 → 30).  
  - Try a different sampler; some samplers produce cleaner details at the same step count.  

- **VRAM out‑of‑memory errors**  
  - Lower base resolution or switch from square to a slightly rectangular aspect ratio.  
  - Use **16‑bit / half‑precision** options if available in your ComfyUI build.  
  - Make sure you’re not accidentally using large batch sizes or extra high‑res passes.

## Wrapping up

With **ComfyUI**, **FLUX.1 Dev**, **T5XXL & CLIP encoders**, and **Real‑ESRGAN**, you have a complete, production‑grade pipeline for generating high‑quality AI art:

- Design your **workflow** visually with nodes.  
- Craft **strong prompts and negative prompts**.  
- Render at a reasonable base resolution, then **upscale to 4K**.  

From here, you can explore more advanced techniques: ControlNet for pose guidance, LoRA for custom styles, or even multi‑pass workflows for ultra‑polished illustrations. I’ll try to cover some of those in future posts—until then, have fun creating your own gallery of AI‑generated masterpieces! 🎨✨
