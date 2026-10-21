# How NVIDIA's DLSS Models Work, and How Games Use Them

*Claims were checked against NVIDIA's own announcements and developer docs. Items still marked [VERIFY] need a final check before publishing.*

## 1. The problem DLSS solves

Real-time rendering is a budget. At 4K, a game has roughly 8.3 million pixels to shade each frame, and ray-traced lighting makes every one of them expensive. Two shortcuts follow from that:

- Render fewer pixels and reconstruct the rest (upscaling).
- Render fewer frames and synthesize the rest (frame generation).

NVIDIA's DLSS family is a set of neural networks that do both, running on the GPU's Tensor Cores. Each model answers a different question:

| Feature | Question it answers | Runs on |
|---|---|---|
| Super Resolution (SR) | Given a low-res frame, what does the full-res frame look like? | All RTX GPUs |
| Ray Reconstruction (RR) | Given a noisy ray-traced image, what is the clean lighting? | All RTX GPUs |
| Frame Generation (FG) | What does the screen look like between two real frames? | RTX 40 and 50 series |
| Multi Frame Generation (MFG) | Same, but for several frames at once | RTX 50 series (see section 5) |
| Neural Rendering ("DLSS 5") | How should lighting and materials look, given the game's 3D data? | See section 6 |

![One frame through a DLSS-enabled engine](figures/fig1-pipeline.svg)

## 2. Super Resolution: the temporal upscaler

DLSS does not upscale a single frame. It accumulates detail across frames.

Each frame, the engine renders at a lower resolution, such as 1080p for a 4K output. It also shifts the camera by a sub-pixel amount, called jitter. Over several frames, every output pixel gets sampled from slightly different positions. The network then combines three things:

1. **Current frame color** at render resolution.
2. **History**: the previous output, reprojected to the current frame.
3. **Motion vectors and depth**: where each pixel moved since the last frame.

![Sub-pixel jitter accumulates samples across frames](figures/fig2-jitter.svg)

The output is a full-resolution frame. Accurate motion vectors are what let the network throw away stale history when an object moves. NVIDIA's guidance calls bad motion vectors the most common cause of visual artifacts.

![Motion vectors reproject history to the current frame](figures/fig3-motion-vectors.svg)

**Architecture.** DLSS 2 through 3.x used convolutional networks (CNNs). DLSS 4, announced at CES in January 2025, moved Super Resolution, Ray Reconstruction, and DLAA to a Transformer model. NVIDIA's claim is that it is the first real-time transformer application in graphics. The stated gains are better stability in motion, less ghosting, and more detail. NVIDIA has not published layer counts or parameter sizes for the Transformer.

**Presets.** The network ships in several "presets" trained for different quality modes. DLSS 4.5 added a second-generation Transformer with about 5x the compute. Its presets are labelled with letters, and NVIDIA documents the mapping for its recommended setting:

- **Preset M:** optimized for DLSS Performance mode.
- **Preset L:** optimized for 4K DLSS Ultra Performance mode.
- **Preset K:** used for the remaining modes. It is the DLSS 4 Transformer model with a minor refinement.

## 3. Ray Reconstruction: replacing the denoiser

Ray tracing is expensive, so games trace only a few rays per pixel. The result is a noisy image. Traditionally each lighting effect (reflections, shadows, global illumination) had its own hand-tuned denoiser. Those denoisers either smear detail over time, which causes ghosting, or blur it spatially.

Ray Reconstruction replaces that stack with a single network trained on high-quality ray-traced reference images. It takes the noisy radiance buffers, plus the same color, motion vector, and depth inputs as SR, and outputs a clean image at full resolution. In effect it denoises and upscales in one pass.

NVIDIA says the DLSS 3.5 Ray Reconstruction model was trained on five times more data than DLSS 3. DLSS 4.5 brought a second-generation Transformer to Ray Reconstruction, with 35% more compute capability, 20% more parameters, a larger training set, and finer developer control over temporal accumulation.

## 4. Frame Generation: synthesizing frames

FG adds frames between two rendered frames. The inputs are the final color, depth, and motion vectors of the real frames, plus an optical flow field that estimates how each region of the image moved between them.

- **DLSS 3 (2022):** the optical flow came from dedicated hardware on RTX 40 cards. Frame Generation was limited to RTX 40.
- **DLSS 4 (2025):** NVIDIA replaced the hardware optical flow with an AI optical-flow model. Frame Generation became faster and used less VRAM, and it moved to RTX 40 and 50 cards.

![Frame generation timeline: 2x and 4x](figures/fig4-frame-generation.svg)

**Interface problem.** Generated frames need to show the HUD correctly, but a HUD has no real motion vectors. To handle this, the game passes a *hudless* buffer, which is the scene before UI is drawn. The network interpolates the scene only. The game then composites the UI back on top. Without the hudless buffer, menus and health bars smear between generated frames.

![Hudless buffer and UI recomposition](figures/fig5-hudless.svg)

**Latency.** Generated frames are not free: they add delay between input and display. NVIDIA Reflex is the mitigation. It paces CPU work so the GPU render queue stays nearly empty, and it can also re-warp the final image with the latest mouse input before scanout.

## 5. Multi Frame Generation

MFG generates up to three frames per rendered frame in its 4x mode. The model runs once per rendered frame and emits several interpolated frames.

NVIDIA restricts MFG to RTX 50 series, and it cites two requirements: the Blackwell Tensor Cores and **Flip Metering**, a display-engine feature that paces frames consistently. Pacing is the hard part, because uneven frame timing feels worse than a lower frame rate. [VERIFY: the claim that Tensor Core throughput is not the limit comes from a third-party thesis, not NVIDIA.]

DLSS 4.5 added Dynamic Multi Frame Generation and a 6x mode, announced at CES 2026. [VERIFY: confirm the hardware list for these two modes on NVIDIA's DLSS 4.5 page before publishing.]

## 6. Neural Rendering ("DLSS 5")

NVIDIA announced DLSS 5 at GTC on March 16, 2026, under the name "3D-Guided Neural Rendering." Per NVIDIA's press release, it takes a game's color and motion vectors for each frame as input, and uses an AI model to infuse the scene with photoreal lighting and materials.

NVIDIA lists DLSS 5 as an RTX 50 Series feature. It has not announced RTX 40 support. Running neural rendering on an RTX 40 card, as some community projects do, is unofficial. It is not a supported NVIDIA configuration and should be described that way.

## 7. How a game uses these models

Games do not call DLSS directly. NVIDIA's **Streamline** SDK is an open-source middleware layer that exposes DLSS SR, FG, RR, Reflex, and latency markers behind one API. The underlying models are shipped as closed-source NGX libraries, such as `nvngx_dlss.dll` and `nvngx_dlssg.dll`.

Each frame, the engine does roughly this:

1. Render the scene with sub-pixel jitter and produce motion vectors, depth, and exposure.
2. Tag these buffers with Streamline.
3. Run DLSS SR (or RR) **before tonemapping**, because the network expects linear HDR light.
4. Draw the UI at full output resolution, after upscaling. UI pixels have no valid motion vectors, so upscaling them would smear the UI.
5. Tag the hudless color and UI mask, then run Frame Generation at present time.
6. Sleep the CPU just in time with Reflex to reduce latency.

Mistakes at any step show up on screen: missing jitter gives flicker, wrong motion vectors give ghosting, and a missing hudless buffer gives UI smear.

## 8. Why injectors conflict

Community tools such as ReShade and OptiScaler work by replacing system DLLs. For example, ReShade is copied in as `dxgi.dll`, so the game loads it instead of the real DirectX library. It forwards calls to the real DLL and hooks the swap chain so post-processing runs every frame.

Three conflicts follow:

- Only one DLL can occupy `dxgi.dll`, so two proxy tools cannot both be installed in the same folder.
- Every injector detours the same functions, so the order of hooks decides which one wins. This can cause crashes or infinite recursion.
- Two tools that modify motion vectors or UI buffers break DLSS-G's assumptions. The frames then look wrong or the HUD smears.

This is why our setup for each game needed one proxy layer, and why we had to disable overlapping frame-generation paths.

![DLL proxy stacking and injector conflicts](figures/fig6-injector-stack.svg)

## 9. Practical takeaways

- Motion vectors and jitter matter more than model version for image quality.
- Frame generation needs a clean hudless buffer and Reflex. Without them, FG looks broken even when it runs.
- Injected tools are fragile because they replace the same system DLLs. Use one proxy layer per game.
- Check any model letter, date, or GPU list against NVIDIA's own notes before publishing.

## Sources to cite

- NVIDIA Streamline SDK: github.com/NVIDIA-RTX/Streamline (programming guides for DLSS, DLSS-G, Reflex)
- NVIDIA DLSS Programming Guide: github.com/NVIDIA/DLSS
- NVIDIA GeForce news: DLSS 4 (Jan 2025), DLSS 3.5 Ray Reconstruction (Aug 2023), DLSS 4.5 (CES 2026)
- NVIDIA press release: DLSS 5 / 3D-Guided Neural Rendering (GTC, Mar 16, 2026): https://nvidianews.nvidia.com (search "DLSS 5")
- NVIDIA DLSS 4.5 Super Resolution: https://www.nvidia.com/en-us/geforce/news/dlss-4-5-super-resolution-available-now/
- NVIDIA DLSS 3.5 Ray Reconstruction: https://www.nvidia.com/en-us/geforce/news/nvidia-dlss-3-5-ray-reconstruction/
- Author background: Gyanateet Dutta, ryukijano.github.io. He is an ML researcher with GPU and neural-rendering work, not a DLSS engineer, so describe him that way.
