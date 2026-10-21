# Inside the Modern NVIDIA DLSS Stack: Architecture, Neural Models, and Engine Integration

An end-to-end technical deep dive into every pillar of the modern NVIDIA neural rendering and latency stack—from **DLSS Super Resolution (SR)** and **Ray Reconstruction (RR)** to **Frame Generation (FG)**, **Multi Frame Generation (MFG 4x & Dynamic 6x)**, **NVIDIA Reflex 2 (Frame Warp)**, and **DLSS 5 (3D-Guided Neural Rendering)**—along with how game engines wire them up through **NVIDIA Streamline / NGX** and why proxy-DLL injectors collide at runtime.

> **Read the full article:** [`post.md`](post.md)

---

## Complete Technology Matrix

| Pillar | Generation & Milestone | Neural / Hardware Architecture | Active Presets & Modes | Official GPU Support |
|---|---|---|---|---|
| **Super Resolution (SR) & DLAA** | DLSS 2/3 (2020–2024)<br>DLSS 4 (Jan 2025)<br>DLSS 4.5 (Jan 2026) | Temporal CNN (6 yrs) $\rightarrow$ 1st-Gen Vision Transformer (2× params, 4× compute) $\rightarrow$ **2nd-Gen Transformer** (5× compute of Gen 1, linear-space training/inference, FP8 Tensor Core acceleration) | **Preset M:** Performance<br>**Preset L:** 4K Ultra Perf<br>**Preset K:** Quality, Balanced, DLAA<br>*(Legacy CNN: Presets A–F)* | **All GeForce RTX GPUs** *(FP8 throughput doubled on RTX 40 / 50 Series)* |
| **Ray Reconstruction (RR)** | DLSS 3.5 (Aug 2023)<br>DLSS 4 (Jan 2025)<br>DLSS 4.5 (Aug/Sep 2026) | Replaces separate hand-tuned GI, reflection, and shadow denoisers with a unified supercomputer-trained neural network (5× data in 3.5 $\rightarrow$ **2nd-Gen Transformer** in 4.5 with +35% compute & +20% parameters) | **Preset F:** DLSS 4.5 2nd-Gen Transformer<br>**Preset D / J:** DLSS 4 1st-Gen Transformer<br>**Preset E:** DLSS 3.5 CNN | **All GeForce RTX GPUs** |
| **Frame Generation (FG)** | DLSS 3 (Sep 2022)<br>DLSS 4 Update (Jan 2025) | DLSS 3 used hardware Optical Flow Accelerator (OFA) + CNN. DLSS 4 replaced hardware OFA with an **AI optical-flow model** (40% faster, 30% less VRAM) | **2x Mode** (1 generated frame per rendered frame; requires Hudless + UI buffers + Reflex) | **GeForce RTX 40 & RTX 50 Series** |
| **Multi Frame Generation (MFG)** | DLSS 4 (Jan 2025)<br>DLSS 4.5 (Jan/Mar 2026) | Split AI optical-flow + synthesis network running once per rendered frame, paired with hardware **Flip Metering** in the Blackwell display engine | **4x Mode:** up to 3 generated frames/real<br>**6x Mode:** up to 5 generated frames/real<br>**Dynamic MFG:** auto-scales 1x–6x to cap refresh rate | **GeForce RTX 50 Series** *(RTX 40 community wrappers are unofficial)* |
| **NVIDIA Reflex & Reflex 2** | Reflex 1 (Sep 2020)<br>Reflex 2 (Jan 2025) | **Reflex 1:** Just-in-time CPU submission eliminating GPU render queue (~50% latency cut).<br>**Reflex 2:** Adds **Frame Warp** (late camera reprojection right before scanout) + **Predictive Inpainting** using prior-frame camera, color, and depth (~75% latency cut) | **Reflex Low Latency:** On / On + Boost<br>**Reflex 2:** Low Latency + Frame Warp | **Reflex 1:** GeForce GTX 900+<br>**Reflex 2 Frame Warp:** RTX 50 Series first; other RTX GPUs planned |
| **Neural Rendering ("DLSS 5")** | Announced GTC (Mar 2026)<br>Launched Fall 2026 | **3D-Guided Neural Rendering**: generative lighting & material model conditioned on per-frame color, motion vectors, depth, and 3D G-buffer anchors; deterministic and temporally coherent up to 4K | Developer controls for **intensity**, **subsurface/material masking**, and **color grading** via Streamline | **GeForce RTX 50 Series** (& GeForce NOW; RTX 40 tuning planned by NVIDIA; RTX 40 mod wrappers are unofficial) |

---

## Visual Architecture Gallery (SVGs & Animated GIFs)

All diagrams and animations in [`figures/`](figures/) are original, self-contained assets generated directly from the scripts in this repository. They are organized below by the subsystem they explain.

### 1. End-to-End Engine Pipeline & Middleware (`figures/fig1-pipeline.svg`)

How a single frame travels through a modern game engine: sub-pixel jittered rendering at lower internal resolution, buffer tagging via **NVIDIA Streamline**, **DLSS Super Resolution / Ray Reconstruction** in linear HDR space *before* tonemapping, native-resolution UI rendering, and display presentation.

![One frame through a DLSS-enabled engine](figures/fig1-pipeline.svg)

---

### 2. DLSS Super Resolution & Ray Reconstruction (`figures/fig2-jitter.svg`, `figures/fig3-motion-vectors.svg`, `figures/anim-jitter-accumulation.gif`)

DLSS Super Resolution and Ray Reconstruction do not upscale a single frame in isolation—they accumulate sub-pixel samples across time and align historical detail using per-pixel motion vectors and self-attention across the frame.

#### Animated: Sub-Pixel Jitter & Temporal Accumulation (`figures/anim-jitter-accumulation.gif`)

![Animated sub-pixel jitter accumulation across four frames](figures/anim-jitter-accumulation.gif)

#### Static Vector Diagrams: Jitter Grid & Motion-Vector Reprojection

| Sub-Pixel Camera Jitter (`figures/fig2-jitter.svg`) | Motion-Vector Reprojection (`figures/fig3-motion-vectors.svg`) |
|---|---|
| ![Sub-pixel jitter accumulates samples across frames](figures/fig2-jitter.svg) | ![Motion vectors reproject history to the current frame](figures/fig3-motion-vectors.svg) |

- **Why linear HDR matters (DLSS 4.5):** Earlier models compressed inputs into log space to suppress fireflies, which muted highlights and shadow contrast. DLSS 4.5 trains and infers directly in linear HDR light with 5× the compute of the first-generation Transformer.
- **Why Ray Reconstruction bypasses separate denoisers:** Instead of running independent hand-tuned spatio-temporal denoisers for global illumination, reflections, and shadows (which blur fine detail or ghost), Ray Reconstruction (`nvngx_dlssd.dll`, **Preset F** in DLSS 4.5) takes raw noisy ray-traced radiance + G-buffer inputs and performs joint denoising and upscaling in one Transformer pass.

---

### 3. Frame Generation, Multi Frame Generation (2x / 4x / 6x), and Hudless UI (`figures/fig4-frame-generation.svg`, `figures/fig5-hudless.svg`, `figures/anim-mfg-pacing.gif`)

Frame Generation synthesizes complete intermediate frames between rendered frames. DLSS 4 replaced the RTX 40 hardware Optical Flow Accelerator (OFA) with a fast AI optical-flow model and introduced **4x Multi Frame Generation**, while DLSS 4.5 added **6x Mode** and **Dynamic Multi Frame Generation** alongside hardware **Flip Metering** in the display engine.

#### Animated: Real-Time Scanout Pacing Across 1x, 2x, 4x, and 6x (`figures/anim-mfg-pacing.gif`)

![Animated comparison of Native, 2x FG, 4x MFG, and 6x MFG frame pacing](figures/anim-mfg-pacing.gif)

#### Static Vector Diagrams: Timeline Multipliers & Hudless UI Recomposition

| 2x, 4x, and 6x Frame Timelines (`figures/fig4-frame-generation.svg`) | Hudless Buffer & UI Recomposition (`figures/fig5-hudless.svg`) |
|---|---|
| ![Frame generation timeline: 2x, 4x, and 6x](figures/fig4-frame-generation.svg) | ![Hudless buffer and UI recomposition](figures/fig5-hudless.svg) |

- **The HUD Problem:** Crosshairs, health bars, and subtitles have zero scene motion vectors. Games pass a **hudless scene buffer** (`nvngx_dlssg.dll` interpolates the 3D world only) and a **UI buffer/alpha mask** so the interface is composited crisply on top of each generated frame.

---

### 4. NVIDIA Reflex & Reflex 2 Frame Warp (`figures/fig7-reflex2-frame-warp.svg`, `figures/anim-reflex2-frame-warp.gif`)

Because Frame Generation holds back a rendered frame to interpolate between two endpoints, raw frame generation adds input latency. **NVIDIA Reflex** is mandatory alongside FG/MFG to keep total system latency low, and **Reflex 2** pushes responsiveness further with **Frame Warp**.

#### Animated: Reflex 2 Late Input Sampling, Frame Warp & Predictive Inpainting (`figures/anim-reflex2-frame-warp.gif`)

![Animated walkthrough of NVIDIA Reflex 2 Frame Warp and Predictive Inpainting](figures/anim-reflex2-frame-warp.gif)

#### Static Vector Diagram: Reflex 2 Four-Stage Pipeline (`figures/fig7-reflex2-frame-warp.svg`)

![NVIDIA Reflex 2 Frame Warp and Predictive Inpainting pipeline](figures/fig7-reflex2-frame-warp.svg)

1. **Reflex 1 (Low Latency Mode):** Synchronizes the CPU with the GPU so simulation and render submission occur *just in time*, eliminating the GPU render queue (~50% average latency reduction).
2. **Reflex 2 (Frame Warp):** While the GPU finishes rendering Frame $N$ at camera pose $t_0$, the CPU samples the latest mouse/controller input and computes the updated camera pose $t_1$. Right before display scanout, Frame Warp shifts the rendered image to $t_1$.
3. **Latency-Optimized Predictive Inpainting:** Warping the image exposes disocclusion gaps along screen edges and behind foreground objects. Reflex 2 fills those gaps using prior-frame camera, color, and depth buffers, cutting end-to-end PC latency by **up to 75%** (e.g., 56 ms Native $\rightarrow$ 27 ms Reflex 1 $\rightarrow$ 14 ms Reflex 2 in *THE FINALS* at 4K).

---

### 5. DLSS 5: 3D-Guided Neural Rendering (`figures/fig8-dlss5-neural-rendering.svg`)

Announced at GTC (March 2026), **DLSS 5** introduces **3D-Guided Neural Rendering**. Unlike unconstrained video diffusion models that hallucinate geometry over time, DLSS 5 is anchored to the game engine's structured 3D buffers—consuming per-frame rendered color, motion vectors, depth, and surface attributes to synthesize photorealistic lighting and material response (subsurface skin scattering, hair sheen, fabric micro-detail) at up to 4K in real time.

![DLSS 5 3D-Guided Neural Rendering architecture](figures/fig8-dlss5-neural-rendering.svg)

---

### 6. Proxy-DLL Injection & Hook Conflicts (`figures/fig6-injector-stack.svg`)

Why do community tools like **ReShade**, **OptiScaler**, **RTXMFG**, and **Steam Overlay** sometimes conflict or crash when stacked in the same game directory? Because they intercept the same DirectX/DXGI entry points (`dxgi.dll`, `winmm.dll`, `IDXGISwapChain::Present`, and Streamline interposer calls).

![DLL proxy stacking and injector conflicts](figures/fig6-injector-stack.svg)

---

## Repository Layout

```text
.
├── post.md                                  # Full technical article (Markdown)
├── README.md                                # Overview, comparison matrix, and visual gallery
├── figures/
│   ├── fig1-pipeline.svg                    # §1: Per-frame engine & DLSS pipeline
│   ├── fig2-jitter.svg                      # §2: Sub-pixel camera jitter grid
│   ├── fig3-motion-vectors.svg              # §2: Motion-vector history reprojection
│   ├── fig4-frame-generation.svg            # §4–5: 2x FG, 4x MFG, and 6x Dynamic MFG timelines
│   ├── fig5-hudless.svg                     # §4: Hudless scene interpolation & UI recomposition
│   ├── fig6-injector-stack.svg              # §9: Proxy-DLL & swap-chain hook conflicts
│   ├── fig7-reflex2-frame-warp.svg          # §6: NVIDIA Reflex 2 Frame Warp & Inpainting
│   ├── fig8-dlss5-neural-rendering.svg      # §7: DLSS 5 3D-Guided Neural Rendering
│   ├── anim-jitter-accumulation.gif         # Animated: 4-frame sub-pixel jitter accumulation
│   ├── anim-mfg-pacing.gif                  # Animated: 1x / 2x / 4x / 6x Flip Metering scanout
│   └── anim-reflex2-frame-warp.gif          # Animated: Reflex 2 late warp & hole inpainting
├── scripts/
│   ├── generate-gifs.ps1                    # Generates the looping GIFs using built-in Windows .NET
│   └── render-figures.ps1                   # Exports SVGs to high-DPI PNGs for blog platforms
└── .gitignore
```

---

## Reproducing the Visuals (GIFs, SVGs, and PNGs)

### 1. Regenerate the Animated GIFs (No External Dependencies)

`scripts/generate-gifs.ps1` uses built-in Windows `.NET` (`System.Drawing` and WPF `PresentationCore` `GifBitmapEncoder`) to synthesize the looping GIFs directly—no third-party tools required:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\generate-gifs.ps1
```

### 2. Render High-DPI PNGs from the SVGs

For publishing platforms (like Medium or Substack) that require raster PNGs instead of `.svg`, install any one of these converters:

| Tool | Windows Install Command |
|---|---|
| **rsvg-convert** | MSYS2: `pacman -S mingw-w64-x86_64-librsvg` |
| **Inkscape** | `winget install --id Inkscape.Inkscape` |
| **ImageMagick** | `winget install --id ImageMagick.ImageMagick` |

Then run from the repository root:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\render-figures.ps1
```

PNG exports are written to `figures/png/` (ignored by git).

---

## Primary Sources & Technical Verification

Every architectural claim in [`post.md`](post.md) and this README was cross-checked against official NVIDIA documentation and research publications, with community/unofficial behavior explicitly separated:

- **DLSS 4 & Multi Frame Generation (CES 2025):** [NVIDIA GeForce News](https://www.nvidia.com/en-us/geforce/news/dlss4-multi-frame-generation-ai-innovations/) & [NVIDIA Applied Deep Learning Research (ADLR) DLSS 4 Page](https://research.nvidia.com/labs/adlr/DLSS4/)
- **DLSS 4.5 Super Resolution, 2nd-Gen Transformer & Dynamic 6x MFG (CES 2026):** [DLSS 4.5 Announcement](https://www.nvidia.com/en-us/geforce/news/dlss-4-5-dynamic-multi-frame-gen-6x-2nd-gen-transformer-super-res/) | [DLSS 4.5 SR Deep Dive](https://www.nvidia.com/en-us/geforce/news/dlss-4-5-super-resolution-available-now/) | [Dynamic MFG & 6x Release](https://www.nvidia.com/en-us/geforce/news/dlss-4-5-dynamic-multi-frame-generation-6x-mode-released/)
- **DLSS 3.5 & DLSS 4.5 Ray Reconstruction:** [DLSS 3.5 RR Announcement](https://www.nvidia.com/en-us/geforce/news/nvidia-dlss-3-5-ray-reconstruction/) | [DLSS 4.5 RR Gamescom Update](https://www.nvidia.com/en-us/geforce/news/gamescom-2026-dlss-4-5-ray-reconstruction-release-announcements-trailers/)
- **NVIDIA Reflex & Reflex 2 (Frame Warp):** [Reflex 2 with Frame Warp Announcement](https://www.nvidia.com/en-us/geforce/news/reflex-2-even-lower-latency-gameplay-with-frame-warp/) | [NVIDIA Reflex Developer Page](https://developer.nvidia.com/performance-rendering-tools/reflex) | [NVIDIA Research Late-Warp Paper](https://research.nvidia.com/publication/2020-07_post-render-warp-late-input-sampling-improves-aiming-under-high-latency)
- **DLSS 5 (3D-Guided Neural Rendering, GTC 2026):** [DLSS 5 GeForce Announcement](https://www.nvidia.com/en-us/geforce/news/dlss5-breakthrough-in-visual-fidelity-for-games/) | [NVIDIA Investor Press Release](https://investor.nvidia.com/news/press-release-details/2026/NVIDIA-DLSS-5-Delivers-AI-Powered-Breakthrough-in-Visual-Fidelity-for-Games/default.aspx) | [NVIDIA Research DLSS 5 Page](https://research.nvidia.com/labs/adlr/DLSS5/)
- **SDKs & Engine Integration Guides:** [NVIDIA Streamline SDK](https://github.com/NVIDIA-RTX/Streamline) | [NVIDIA DLSS Programming Guide](https://github.com/NVIDIA/DLSS)

## License

All Markdown prose, SVG diagrams, and generated GIF animations in this repository are original works. Do not add copyrighted NVIDIA marketing screenshots or proprietary binaries without permission.
