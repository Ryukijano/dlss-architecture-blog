# NVIDIA's Neural Rendering Stack in October 2026: What Ships, What Is Announced, and Where It Is Heading

*Status date: 22 October 2026. Claims are checked against NVIDIA's GeForce news posts, the ADLR research pages, and NVIDIA's support matrix. Where a claim comes from third-party reporting, or from a personal test on my own PC, the text says so.*

---

## 0. Summary

NVIDIA's neural rendering stack now covers four jobs: rebuilding pixels (Super Resolution), cleaning up ray-traced light (Ray Reconstruction), inventing frames (Frame Generation and Multi Frame Generation), and, since September 2026, generating the final lighting and materials (DLSS 5). Latency is handled by Reflex, and Reflex 2's late camera warp is the feature still missing from shipping products.

Three points matter most for anyone deciding what to buy or enable:

1. **Hardware gates are getting stricter, not looser.** Super Resolution and Ray Reconstruction run on every GeForce RTX card. Multi Frame Generation and DLSS 5 are officially RTX 50 only. RTX 40 owners get DLSS 5 only through unofficial ports so far.
2. **The ms budget is the real constraint.** At 240 Hz, one frame has about 4.2 ms. Every neural pass has to fit into that slot, which is why NVIDIA spends so much effort on model size, FP8 math, and display-side pacing.
3. **Some announcements are ahead of shipping.** Reflex 2 Frame Warp was announced in January 2025 and is still labeled "coming soon" on NVIDIA's own page in October 2026.

---

## 1. Status Snapshot (22 October 2026)

| Technology | Status | Official hardware | Evidence |
|---|---|---|---|
| Super Resolution, 2nd-gen Transformer (DLSS 4.5) | Shipping since 14 Jan 2026 | All GeForce RTX | NVIDIA app rollout, 400+ games and apps |
| Ray Reconstruction, 2nd-gen Transformer (DLSS 4.5, Preset F) | Shipping since 25 Aug 2026 | All GeForce RTX | NVIDIA Gamescom post |
| Frame Generation (2x) | Shipping since 2022; model updated in 2025 | RTX 40 and RTX 50 | NVIDIA DLSS 4 post |
| Multi Frame Generation, 4x | Shipping since Jan 2025 | RTX 50 only | Over 250 games and apps (NVIDIA, Jan 2026) |
| Dynamic MFG and 6x MFG | Rolled out via NVIDIA app on 31 Mar 2026 | RTX 50 only | Tom's Hardware called this a beta; NVIDIA's post calls it available now |
| Reflex 1 (Low Latency) | Shipping since 2020 | GeForce GTX 900 and newer | NVIDIA Reflex page |
| Reflex 2 with Frame Warp | **Announced Jan 2025, not shipping** | RTX 50 first, other RTX later | NVIDIA page still says "coming soon"; no shipping game found |
| DLSS 5, 3D-Guided Neural Rendering | Launched Sept 2026 in NBA 2K27; October model update | RTX 50 only (plus GeForce NOW and RTX Spark laptops) | NVIDIA post, Oct 13 2026 |

![One frame through a DLSS-enabled engine](figures/fig1-pipeline.svg)

---

## 2. The Budget Problem

A 4K frame has 3840 × 2160 ≈ 8.3 million pixels. Ultrawide 7680 × 2160 has about 16.6 million. Path tracing shades each of them with only one or two rays. Three numbers frame the design space:

- **60 Hz:** 16.7 ms per frame.
- **240 Hz:** 4.2 ms per frame.
- **Frame Generation at 4x on a 240 Hz panel:** each generated frame has to be produced in roughly 1 ms, or it eats the time needed for the next real frame.

NVIDIA's own framing reflects this. DLSS 4.5 "draws 23 out of every 24 pixels" on screen with AI, and DLSS 4 Multi Frame Generation "generates 15 out of every 16 pixels" (NVIDIA claims, Jan 2025 and Jan 2026). Those numbers are a useful way to think about the stack: the GPU renders a small fraction of what you see, and the models fill in the rest.

NVIDIA's ADLR team describes the MFG timing concretely. DLSS 3 Frame Generation produced one 4K frame in about 3.25 ms on an RTX 4090. DLSS 4 MFG produced each of its three frames in about 1 ms on an RTX 5090 at launch. Those are different GPUs and different generations, so the comparison is a trend, not a like-for-like benchmark.

---

## 3. Super Resolution: From Temporal CNNs to Transformers

### 3.1 The temporal loop

Super Resolution is a temporal accumulator, not a per-frame upscaler. Each frame the engine:

1. Renders at a lower internal resolution. For 4K Performance mode that is 1080p. Ultra Performance uses 720p.
2. Shifts the projection by a sub-pixel **jitter** offset, so that each frame samples a different point inside each output pixel.
3. Writes a **motion vector** for each pixel, which says where that surface was in the previous frame.

![Sub-pixel jitter accumulates samples across frames](figures/fig2-jitter.svg)

The network then combines the current low-resolution color, the previous high-resolution output reprojected with motion vectors, and depth and exposure. The jitter ensures that over several frames the history contains samples that the current frame lacks.

![Motion vectors reproject history to the current frame](figures/fig3-motion-vectors.svg)

![Animated sub-pixel jitter accumulation across four frames](figures/anim-jitter-accumulation.gif)

*Animation note: the GIF is a schematic of the jitter pattern. It is not captured from a game.*

### 3.2 Where motion vectors fail

NVIDIA's ADLR team says motion vectors are "accurate for many pixels" but "inaccurate for specular highlights, reflections, and UI motion." Depth is wrong for lasers and for UI drawn after rasterization. These are exactly the areas that viewers notice most, even though they are a small share of pixels. This is the main reason the network needs learned priors and not only geometry.

### 3.3 Model generations

- **DLSS 2 and 3.x (2020–2024):** a generalized temporal CNN, with Presets A to F. NVIDIA says that after six years of CNN work it "reached the limits of what's possible with the DLSS CNN architecture."
- **DLSS 4 (January 2025):** the first real-time vision Transformer used for Super Resolution, Ray Reconstruction, and DLAA. NVIDIA says it has **double the parameters** of the CNN and uses self-attention across the frame and over time. The stated benefits are stability, less ghosting, more detail in motion, and smoother edges.
- **DLSS 4.5 (January 2026):** a second-generation Transformer. NVIDIA says it uses **five times the compute** of the first-generation Transformer and was trained on a significantly expanded dataset.

### 3.4 Presets you will see

The NVIDIA app's DLSS override offers a "Recommended" setting, which maps:

- **Preset M:** DLSS Performance.
- **Preset L:** DLSS Ultra Performance at 4K.
- **Preset K:** Quality, Balanced, and DLAA.

NVIDIA also notes that RTX 20 and RTX 30 cards lack native FP8 support, so M and L cost more there. Those users may prefer K. The NVIDIA app's statistics overlay (Alt+Z, then Statistics, then DLSS) shows which model is active.

---

## 4. Ray Reconstruction: Replacing the Denoiser Stack

Ray tracing at one or two samples per pixel gives a noisy image. Engines used to run separate hand-tuned denoisers for global illumination, reflections, and shadows. Each one smeared detail over time or blurred it in space.

Ray Reconstruction replaces that stack with one network. NVIDIA describes it as a model that "unifies denoising and Super Resolution into a single model," which uses temporal and spatial engine data to rebuild the high-resolution image.

- **DLSS 3.5 (August 2023):** introduced the model, trained on five times the data of DLSS 3.
- **DLSS 4 (January 2025):** moved RR to the Transformer architecture.
- **DLSS 4.5 (25 August 2026):** a second-generation Transformer with **35% more compute capability and 20% more parameters** at similar performance, plus a larger training set and finer developer control over temporal accumulation. The Ray Reconstruction menu offers "Recommended" or "Preset F."

---

## 5. Frame Generation, Multi Frame Generation, and the HUD Problem

### 5.1 What frame generation does

Frame Generation inserts frames between two real frames. It uses the real frames' color, depth, and motion vectors, plus an optical-flow field that estimates motion where vectors are missing.

- **DLSS 3 (2022):** relied on the RTX 40 **Optical Flow Accelerator** hardware to build that flow field, and produced one frame per real frame (2x).
- **DLSS 4 (January 2025):** replaced the hardware optical flow with an **AI optical-flow model**. NVIDIA says the new generation model is 40% faster and uses 30% less VRAM. It runs once per rendered frame, and it works on RTX 40 and RTX 50.

![Frame generation timeline: 2x, 4x, and 6x](figures/fig4-frame-generation.svg)

### 5.2 The multipliers

- **4x MFG (DLSS 4):** up to three generated frames per real frame. NVIDIA says the GPU runs five AI models per rendered frame, all within a few milliseconds, and uses fifth-generation Tensor Cores with up to 2.5 times the AI throughput of the previous generation.
- **6x MFG and Dynamic MFG (DLSS 4.5):** up to five generated frames per real frame. Dynamic MFG changes the multiplier during play to hit a target frame rate or the display's refresh rate. NVIDIA describes it as "an automatic transmission for your GPU."

Both 6x and Dynamic MFG require RTX 50 hardware, and NVIDIA says they require GeForce Game Ready Driver 595.97 WHQL or newer.

### 5.3 Pacing and Flip Metering

Generated frames must arrive at even intervals. Uneven spacing feels worse than a lower frame rate. NVIDIA says DLSS 3 used CPU-side pacing whose variability "can compound with additional frames." For Blackwell, NVIDIA moved pacing into the **display engine** through hardware **Flip Metering**, and doubled the display engine's pixel processing for it.

Flip Metering is an NVIDIA claim about hardware behavior. I have not measured pacing on Blackwell or Ada hardware, so the post does not give a pacing percentage.

![Animated comparison of Native, 2x FG, 4x MFG, and 6x MFG frame pacing](figures/anim-mfg-pacing.gif)

*Animation note: this is a schematic of slot timing, not a capture of real frame times.*

### 5.4 The HUD problem

A crosshair, health bar, or subtitle has no 3D motion. If frame generation interpolated the composited image, UI would smear between frames. The fix is to pass the engine a **hudless** scene buffer, interpolate only the 3D scene, and composite the UI back on top.

NVIDIA's DLSS 4.5 Frame Generation model update uses "additional engine data" to improve static UI. Tom's Hardware reports a new Frame Generation model called "Preset B" that uses additional UI buffers, and that it runs on RTX 40 and RTX 50. I have not confirmed Preset B on NVIDIA's own pages, so I list it only as a third-party report.

![Hudless buffer and UI recomposition](figures/fig5-hudless.svg)

---

## 6. Reflex and Reflex 2

### 6.1 Reflex 1 (shipping)

Reflex makes the CPU wait until the GPU is ready for the next frame. Instead of letting the CPU queue several frames ahead, Reflex submits work just in time. That removes the GPU render queue, and it lets input be sampled later in the frame. NVIDIA says Reflex reduces latency by about 50% on average in competitive games, that it is integrated in over 100 games, and that over 90% of GeForce users turn it on.

Frame generation needs Reflex. Frame generation holds back a real frame so it can interpolate between two real frames, and Reflex offsets some of that wait.

### 6.2 Reflex 2 with Frame Warp (announced, not shipping)

Reflex 2 adds a late camera update to Reflex 1. NVIDIA's description:

1. While the GPU renders frame N, the CPU computes the camera position for the next frame from the latest mouse or controller input.
2. Right before scanout, Frame Warp takes the finished image and warps it to that newer camera position.
3. Warping exposes holes where the camera reveals scene regions the old frame did not contain. NVIDIA fills them with a "latency-optimized predictive rendering algorithm" that uses camera, color, and depth from prior frames.

![Animated walkthrough of NVIDIA Reflex 2 Frame Warp and Predictive Inpainting](figures/anim-reflex2-frame-warp.gif)

*Animation note: an illustration of NVIDIA's described steps. The latency values are NVIDIA's demo numbers, not a measurement of a shipping product.*

![NVIDIA Reflex 2 Frame Warp and Predictive Inpainting pipeline](figures/fig7-reflex2-frame-warp.svg)

NVIDIA's demo numbers, for *THE FINALS* at 4K max settings on an RTX 5070, are 56 ms native, 27 ms with Reflex Low Latency, and 14 ms with Reflex 2. NVIDIA also reports under 3 ms average PC latency in *VALORANT* at 800+ FPS on an RTX 5090. These numbers are NVIDIA's own claims for a demo build.

**Status.** NVIDIA's January 2025 post says Reflex 2 "is coming soon to *THE FINALS* and *VALORANT*," debuts on RTX 50, and will come to other RTX GPUs "in a future update." NVIDIA's Reflex page still says "coming soon" in October 2026. VideoCardz and OC3D reported in mid-2026 that no shipping game includes it, and that NVIDIA had not answered requests for a date. I could not find a shipping title or a release date, so I treat Reflex 2 as **not shipped**.

NVIDIA's ADLR team describes Frame Warp as providing "an alternative trade-off for competitive games where minimizing latency is paramount." My reading is that it attacks the input-latency cost from a different direction than frame generation does, but NVIDIA does not spell out how the two combine.

---

## 7. DLSS 5: 3D-Guided Neural Rendering

### 7.1 What it does

DLSS 5 is the first DLSS technology that generates the final displayed appearance rather than reconstructing a higher-cost reference from the normal renderer, according to NVIDIA's ADLR paper (published 1 September 2026). The model is:

- **One-step pixel-space diffusion.** It produces the output in a single inference step, not a multi-step denoising loop.
- **Conditioned on** the current rendered frame, the engine's motion vectors, a carried temporal state, and artistic-direction values.
- **Trained with consistency supervision** from renderer-derived scene attributes, so the output stays tied to authored geometry and materials.
- **Causal and deterministic**, trained for frame-to-frame stability, and held to a strict per-frame compute budget.

NVIDIA's examples are subsurface scattering in skin, sheen on fabric, light interaction in hair, and front-, back-, and overcast-lit scenes. Developers get controls for intensity, color grading, and masks, and integration uses Streamline, the same SDK as DLSS and Reflex.

![DLSS 5 3D-Guided Neural Rendering architecture](figures/fig8-dlss5-neural-rendering.svg)

### 7.2 The timeline

- **16 March 2026:** announced at GTC. NVIDIA called it "the GPT moment for graphics."
- **September 2026:** launched in *NBA 2K27* on RTX 50 and GeForce NOW.
- **October 2026:** NVIDIA's update reduced the model to about 118 million parameters, down 24%. It also cut 4K VRAM use by 20% and halved GPU overhead relative to the September version, with comparable visual fidelity. NVIDIA says the model is now 10 times faster than when it was announced, and up to 1.4 times faster in frame rate in *NBA 2K27*.
- **Upcoming:** *Hogwarts Legacy* (5 November), *Where Winds Meet*, and RTX Remix (23 October). NVIDIA says the new model ships by default with future DLSS 5 games.

### 7.3 Hardware and the RTX 40 question

NVIDIA's support matrix marks DLSS 3D-Guided Neural Rendering as RTX 50 only. The March announcement did not mention RTX 40. In a statement reported by The Verge and TechPowerUp, NVIDIA said it is focused on RTX 50 optimization, that "model updates [are] expected later this fall," and that "once RTX 50 Series performance is more fully tuned, we plan to work on expanding official support to the GeForce RTX 40 Series." NVIDIA gave no date.

On the RTX 40 side, modders have run a leaked DLSS 5 build and a newer model on RTX 40 cards. Those runs are unofficial, and NVIDIA does not support them.

### 7.4 Controversy

The Verge calls DLSS 5 "controversial," and some players complained that NVIDIA was "artificially locking" it to RTX 50. The question I would watch is not whether it looks good. It is whether a generative stage that changes the look of the scene keeps the developer's intent. NVIDIA's answer is determinism, consistency supervision, and artistic controls. Those are design claims, and they need to be tested per game.

---

## 8. How a Game Uses the Stack

Games do not call the neural networks directly. They use two layers:

1. **NVIDIA Streamline** (`sl.interposer.dll`, `sl.dlss.dll`, `sl.dlss_d.dll`, `sl.dlss_g.dll`, `sl.reflex.dll`) is the open SDK the engine links against. The game tags its buffers and places Reflex markers.
2. **NGX** (`nvngx_dlss.dll`, `nvngx_dlssd.dll`, `nvngx_dlssg.dll`, `nvngx_dlssnr.dll`) is the closed runtime that holds the model weights and presets.

NVIDIA reported over 800 RTX-enhanced games and apps in its January 2026 DLSS 4.5 post. Adoption depends on engines tagging buffers correctly. The model can only be as good as the motion vectors, jitter, depth, and hudless buffer it receives.

Per frame, a correct integration runs roughly in this order:

1. Reflex sleep and input sampling.
2. Jittered rendering at internal resolution, with motion vectors and depth.
3. Super Resolution or Ray Reconstruction.
4. Post-processing and a hudless snapshot.
5. Native-resolution UI pass and UI buffer.
6. Frame generation, Reflex 2 warp if shipping, and presentation.

I did not verify the exact input color space against the Streamline programming guide. Check it there before writing engine code.

---

## 9. Why Community Injectors Conflict

Enthusiast setups often stack ReShade, OptiScaler, and a frame-generation add-on, plus Steam Overlay. These tools rely on **DLL proxying** and **detouring**:

- Windows loads `dxgi.dll` or `winmm.dll` from the game folder before System32. One folder can only hold one proxy under each name.
- ReShade and OptiScaler both wrap the swap chain and hook `IDXGISwapChain::Present`. Hook order decides which one runs first. In my tests, Steam Overlay injected into Cyberpunk and then detached after a few seconds, and one Crimson Desert launch crashed with heap corruption (`0xc0000374`). I did not isolate the cause of the crash, so I do not attribute it to a specific hook.
- Two frame-generation paths on one swap chain can compete for the present queue and the hudless tags. Disabling OptiScaler's frame generation was the fix I used in both games.

![DLL proxy stacking and injector conflicts](figures/fig6-injector-stack.svg)

**Anti-cheat.** Injecting DLLs into multiplayer games can trigger anti-cheat systems, and this is a separate risk from the visual conflicts above. Check each game's policy before you use injectors in online play.

---

## 10. Case Study: One RTX 4090, Two Games (Personal Tests)

These are my observations from my own PC, not NVIDIA results. They are in the log files I inspected, and I did not measure frame timing or visual quality.

**Cyberpunk 2077**
- Native DLSS Frame Generation enabled. Multi Frame Generation set to x4 in the game.
- OptiScaler overrides: Super Resolution Preset M (`RenderPresetForAll = 13`), Ray Reconstruction Preset F (`RenderPresetForAll = 6`).
- The OptiScaler log confirmed both overrides were parsed, and it confirmed DLSSD v310.9.1.
- Streamline reported "multi-frame supported, max generated frames 5."
- **Not verified:** actual x4 output on screen, and frame intervals.

**Crimson Desert**
- Native frame generation enabled. Injected frame generation disabled (`FGInput=nofg`, `FGOutput=nofg`).
- Logs showed native DLSS-G hooking, generated frames, and a maximum of 5 generated frames per real frame.
- One launch crashed with heap corruption, and a later launch ran.
- **Not verified:** pacing, visual quality, and whether Steam Overlay works in a Steam-launched session.

Takeaway from this setup: on an RTX 4090, the official stack is SR, RR, and 2x frame generation. The 4x and 6x modes in these logs came through unofficial routes, and the logs are the only evidence I have for them.

---

## 11. Outlook

### 11.1 What will happen in the next six months

These are dated items from NVIDIA or from confirmed reporting.

- **DLSS 5 on RTX 40:** NVIDIA says the plan follows RTX 50 tuning, with model updates "later this fall." No date for RTX 40.
- **DLSS 5 games:** *Hogwarts Legacy* on 5 November; *Where Winds Meet* and more titles in the coming weeks, per NVIDIA.
- **Reflex 2:** no date. Watch for the SDK, a driver toggle, and a shipping title. Until one of those appears, do not count Frame Warp in a buying decision.
- **Dynamic MFG and 6x:** wider game support as more titles integrate the new overrides.

### 11.2 Structural trends

**From reconstruction to generation.** Super Resolution rebuilds pixels that the GPU rendered. Frame generation invents frames. DLSS 5 generates the lighting and materials of the final image, anchored to the renderer. This is a larger change than any single model update, because it moves some of the artistic result out of the engine and into a model. NVIDIA is betting that developers keep control through the intensity, mask, and grading tools. The next two years will show whether that bet holds.

**Hardware gating will widen.** Super Resolution and Ray Reconstruction are software-portable across generations, and NVIDIA has kept them on all GeForce RTX. Multi Frame Generation and DLSS 5 are the first features that tie to new Tensor Core and display-engine capabilities. The RTX 40 path for DLSS 5 is a useful test of how far NVIDIA is willing to backport a model that is "most computationally demanding to date."

**Models are getting cheaper fast.** According to NVIDIA, the DLSS 5 model went from announcement in March to a 10x speedup by October, with a smaller footprint. That pace is unusual for a generative model. My reading is that the first public version was tuned for a demo budget, but NVIDIA has not said so.

**Latency is the next competitive axis.** Reflex 1 is shipping and widely enabled. Reflex 2 is the next step, but it is delayed. The gap between announcement and shipping is a reminder that engine integration and driver work are the slow parts, not the model.

### 11.3 Risks and open questions

- **Determinism versus perceived change.** NVIDIA says DLSS 5 is deterministic and temporally stable. Players and artists will still judge whether the look matches the game. Per-game tests are the only answer.
- **Frame generation and input.** Frame generation adds latency by design. Reflex limits it, but it does not remove it. Reflex 2 was meant to address this, and it is not shipping.
- **Vendor-specific paths.** MFG, Flip Metering, and DLSS 5 depend on NVIDIA hardware and drivers. Engines that support them will not run them elsewhere.
- **Injected stacks.** Community proxies break when games update and when they collide. They also carry anti-cheat risk. Use one proxy per game, and prefer native features.
- **Unverified claims in this post.** NVIDIA's pacing and latency numbers come from NVIDIA's demos. I have not measured them on my hardware.

### 11.4 What I would do with an RTX 4090 today

- **Super Resolution:** use the NVIDIA app's "Recommended" model preset. Use Preset M for Performance.
- **Ray Reconstruction:** use Preset F or "Recommended" where the game supports it.
- **Frame Generation:** enable native 2x. Leave Reflex on.
- **MFG:** treat 4x and 6x as unsupported on RTX 40, unless you are testing knowingly.
- **DLSS 5:** wait for RTX 40 official support, or test a community port only in offline or single-player settings.
- **Reflex 2:** do not plan around it until NVIDIA ships it.

---

## 12. Sources

NVIDIA primary sources:

1. DLSS 5 announcement (GTC, 16 March 2026): https://www.nvidia.com/en-us/geforce/news/dlss5-breakthrough-in-visual-fidelity-for-games/
2. DLSS 5 October update, model speedup and Hogwarts Legacy date (13 October 2026): https://www.nvidia.com/en-us/geforce/news/dlss-5-faster-ai-model-hogwarts-legacy-rtx-spark-laptops/
3. ADLR, DLSS 5: Generative Neural Rendering (1 September 2026): https://research.nvidia.com/labs/adlr/DLSS5/
4. DLSS 4 Multi Frame Generation and transformers (6 January 2025): https://www.nvidia.com/en-us/geforce/news/dlss4-multi-frame-generation-ai-innovations/
5. ADLR, DLSS 4: https://research.nvidia.com/labs/adlr/DLSS4/
6. DLSS 4.5 Super Resolution (14 January 2026): https://www.nvidia.com/en-us/geforce/news/dlss-4-5-super-resolution-available-now/
7. DLSS 4.5 Ray Reconstruction (25 August 2026): https://www.nvidia.com/en-us/geforce/news/gamescom-2026-dlss-4-5-ray-reconstruction-release-announcements-trailers/
8. DLSS 4.5 Dynamic MFG and 6x (31 March 2026): https://www.nvidia.com/en-us/geforce/news/dlss-4-5-dynamic-multi-frame-generation-6x-mode-released/
9. Reflex 2 with Frame Warp (6 January 2025): https://www.nvidia.com/en-us/geforce/news/reflex-2-even-lower-latency-gameplay-with-frame-warp/
10. DLSS support matrix: https://www.nvidia.com/en-us/geforce/technologies/dlss/
11. Streamline SDK: https://github.com/NVIDIA-RTX/Streamline

Third-party reporting, used only where noted:

- The Verge, RTX 40 statement and "controversial" framing: https://www.theverge.com/games/989978/nvidia-dlss-5-rtx-40
- TechPowerUp, RTX 40 confirmation: https://www.techpowerup.com/352330/nvidia-confirms-dlss-5-for-geforce-rtx-40-series-is-coming
- VideoCardz, Reflex 2 status: https://videocardz.com/newz/nvidia-reflex-2-remains-unreleased-18-months-after-announcement-rtx-owners-still-waiting
- Tom's Hardware, DLSS 4.5 Dynamic MFG beta and Preset B: https://www.tomshardware.com/pc-components/gpus/nvidia-dlss-4-5-dynamic-mfg-plus-5x-and-6x-framegen-modes-enter-beta-for-rtx-50-series-users-update-offers-greater-control-over-generated-frame-rates-and-more-headroom-for-high-refresh-rate-displays
