# Under the Hood: GeForce SMs, Tensor Cores, Optical Flow, and Multi Frame Generation

*Companion to [`post.md`](post.md). Status date: 22 October 2026. Each claim is marked as NVIDIA's statement, third-party analysis, or my inference. NVIDIA does not publish layer counts, token layouts, or parameter counts for most DLSS models, and this document says where that gap matters.*

---

## 1. Why the hardware matters

A DLSS pass is a neural network that has to finish inside a frame budget. At 240 Hz, a frame gets about 4.2 ms, and the GPU also has to render the game. Each generation of GeForce changes three things that decide whether a model fits that budget:

1. **How many tensor operations the SM can run per clock, and in which number formats.** Lower precision means more operations per second and less memory traffic.
2. **Fixed-function blocks that do work outside the SM.** The Optical Flow Accelerator and the display engine are the two that matter for DLSS.
3. **Scheduling and display pacing.** A generated frame that arrives late is worse than no generated frame at all.

---

## 2. The Streaming Multiprocessor, generation by generation

![SM layout across Ampere, Ada, and Blackwell](figures/fig9-sm-generations.svg)

Every GeForce SM in these generations has the same outer structure: four sub-cores, each with a warp scheduler, a slice of the register file, and one tensor core. What changes is the per-sub-core datapath, the tensor core's precision support, and the RT core.

| Feature (per SM) | Ampere (GA102, RTX 30) | Ada (AD102, RTX 40) | Blackwell (GB202, RTX 50) |
|---|---|---|---|
| Compute capability | 8.6 | 8.9 | 12.0 |
| CUDA cores | 128 | 128 | 128 |
| Warp schedulers | 4 | 4 | 4 |
| Per sub-core datapath | 16 FP32 + 16 FP32/INT32 | 16 FP32 + 16 FP32/INT32 | 32 unified FP32 or INT32 |
| Tensor cores | 4, 3rd gen | 4, 4th gen | 4, 5th gen |
| RT cores | 1, 2nd gen | 1, 3rd gen | 1, 4th gen |
| Register file | 256 KB | 256 KB | 256 KB |
| L1 / shared memory | 128 KB | 128 KB | 128 KB |

The sub-core datapath row needs a note. The Ampere and Ada whitepapers describe each sub-core as 16 FP32-only lanes plus 16 lanes that run FP32 or INT32. Blackwell's whitepaper describes a unified datapath, where a sub-core runs either FP32 or INT32 in a given clock. NVIDIA says it designed this for neural shaders, where integer work handles address generation for small neural networks inside a shader.

### Card-level numbers

| Spec | RTX 3090 (GA102) | RTX 4090 (AD102) | RTX 5090 (GB202) |
|---|---|---|---|
| SMs | 82 | 128 | 170 |
| CUDA cores | 10,496 | 16,384 | 21,760 |
| Tensor cores | 328 | 512 | 680 |
| RT cores | 82 | 128 | 170 |
| L2 cache | 6 MB | 72 MB | 96 MB |
| Memory | 24 GB GDDR6X, 936 GB/s | 24 GB GDDR6X, 1,008 GB/s | 32 GB GDDR7, 1,792 GB/s |

These counts come from NVIDIA's spec pages and whitepapers, cross-checked with third-party reviews.

### What changed, generation by generation

- **Ampere to Ada:** the SM count per card rises, and L2 grows from 6 MB to 72 MB on the flagship dies. Tensor cores gain FP8. The RT core is 3rd gen, with opacity micromaps and displaced micro-meshes, which NVIDIA describes as ways to cut ray-tracing memory and work.
- **Ada to Blackwell:** tensor cores gain FP4 (NVFP4) and a second-generation FP8 transformer engine. The datapath is unified. The RT core is 4th gen, with a triangle cluster intersection engine for NVIDIA's Mega Geometry.

Per-SM counts barely change, so most of the RTX 50 throughput gain comes from more SMs, clocks, memory bandwidth, and the newer tensor core, not from a bigger SM.

---

## 3. Tensor cores and number formats

![Tensor core precision support by generation](figures/fig10-tensor-precision.svg)

Tensor cores run matrix multiplies. The number format decides how many operations fit in one clock and how much memory each value takes.

- **FP16 and BF16** work on all three generations and are the base format for older DLSS models.
- **TF32** arrived with Ampere.
- **FP8** arrived with Ada. NVIDIA's ADLR DLSS 4 page says the Transformer models are trained and run in FP8 precision.
- **FP4 (NVFP4)** arrived with Blackwell. NVIDIA states it can "double AI throughput while halving memory requirements" compared with FP8, but DLSS has not been publicly documented as using FP4.
- **Structured sparsity (2:4)** has been in NVIDIA tensor cores since Ampere. It skips zeroed values in a weight matrix.

**Read the headline numbers carefully.** NVIDIA's "AI TOPS" figures use each generation's lowest precision and sparsity, so they are not comparable. RTX 3090 is 285 TFLOPS FP16 sparse. RTX 4090 is 1,321 TOPS FP8 sparse. RTX 5090 is 3,352 TOPS FP4 sparse. At matched precision the generational gain is much smaller.

### What RTX 20 and RTX 30 lack—and what that means

RTX 20 and RTX 30 cards have no native FP8. NVIDIA says DLSS 4.5 Super Resolution's M and L presets therefore cost more on those cards, and recommends Preset K there. For DLSS 5 community ports, lack of FP8 is an efficiency problem, not a proof of impossibility: an FP16-compiled model has been demonstrated on SM86. It still costs much more time per frame than the Ada FP8 path and is not the same official runtime/model combination.

---

## 4. Ray tracing cores (brief)

RT cores accelerate ray-triangle and ray-box tests. They matter for DLSS because Ray Reconstruction only needs to clean up as many rays as the RT cores can trace each frame.

- **2nd gen (Ampere):** about 2x Turing's ray-triangle rate, with hardware for motion blur.
- **3rd gen (Ada):** about 2x Ampere's ray-triangle rate, opacity micromaps, and displaced micro-meshes, which compress geometry for BVH building.
- **4th gen (Blackwell):** a triangle cluster intersection engine for Mega Geometry, which handles clusters of triangles together.

These are NVIDIA's claims from the launch material. I did not measure them.

---

## 5. The Optical Flow Accelerator

![Optical flow: hardware OFA versus AI optical flow](figures/fig11-ofa-pipeline.svg)

### 5.1 What it is

The Optical Flow Accelerator (OFA, also called NVOFA) is a fixed-function block that computes **dense optical flow** between two frames: for each block of pixels, a vector saying where that content moved. NVIDIA's Optical Flow SDK documentation says it runs independently of the CUDA cores. It has been present since Turing.

### 5.2 What the SDK exposes

From NVIDIA's Optical Flow SDK programming guide:

- **Input:** pairs of YUV or RGB frames.
- **Output:** flow vectors per block, at quarter-pixel precision. Block size is 4x4, 2x2, or 1x1 pixels. The 2x2 and 1x1 grids are Ampere and newer features.
- **Hints:** an external flow hint can be supplied, which lets an engine feed in its own motion vectors.
- **Cost output:** an optional per-vector cost, where a higher cost means the vector is less reliable.
- **Presets:** three quality/speed presets.
- **Frame-rate conversion:** NVIDIA also ships a FRUC library built on the same hardware.

NVIDIA's SDK page says the OFA is available on "Turing and newer." The SDK does not say whether a given consumer Blackwell card has the block, and I could not confirm that from NVIDIA's public material.

### 5.3 How DLSS 3 used it

DLSS 3 Frame Generation (RTX 40, September 2022) fed two consecutive rendered frames to the OFA. The OFA's flow field went into a convolutional autoencoder along with the current and prior frames, engine motion vectors, and depth. NVIDIA says the flow field captures "pixel-level information such as particles, reflections, shadows, and lighting" that engine motion vectors miss, while engine vectors stay more accurate for geometry.

### 5.4 Why DLSS 4 replaced it

NVIDIA's DLSS 4 announcement says the DLSS 3 approach made multi-frame generation "cost prohibitive," because the OFA and the AI model both ran for every generated frame. DLSS 4 replaced the OFA with "a very efficient AI model," and the new Frame Generation model runs once per rendered frame. NVIDIA says it is 40% faster and uses 30% less VRAM.

In a Digital Foundry interview, Bryan Catanzaro, NVIDIA's VP of applied deep learning research, explained the reasoning. Hardware optical flow is fixed: once built, it cannot be retrained. Its failures "couldn't be undone with a smarter neural network." A learned flow model can be trained together with the generation network, and it runs on the Tensor cores that are already there.

### 5.5 Where this leaves RTX 50

NVIDIA's public DLSS material does not say whether the OFA block remains on RTX 50 silicon. DLSS 4 does not use it. The hardware question is open, and I do not state it either way.

---

## 6. How Multi Frame Generation works

![Multi Frame Generation: one shared network half, one small half per frame](figures/fig12-mfg-split.svg)

### 6.1 The split network

NVIDIA's ADLR DLSS 4 page describes the MFG architecture as follows:

> "The new DLSS Multi Frame Generation architecture splits the neural component of DLSS 3's Frame Generation in half. One half of the network runs once for every input frame pair and its output is then able to be re-used. The other (much smaller) half runs once for every generated output frame."

So for a 4x mode, the expensive half runs once for the pair of rendered frames (A and B). The small half then runs three times, once for each generated frame. Its output depends on the shared half's result and on a position in time.

### 6.2 Timing the generated frames

NVIDIA does not describe the timing input. A third-party reverse engineering of `nvngx_dlssg.dll` (the mfg-unlock project) reports that the interpolation position is `t = index / (count + 1)`. For 4x that gives t = 1/4, 2/4, and 3/4. For 6x it gives five positions, from 1/6 to 5/6. This matches NVIDIA's description, but it is not an NVIDIA disclosure.

### 6.3 Official modes and community unlocks

| Mode | Official NVIDIA support | Community result |
|---|---|---|
| 2x Frame Generation | RTX 40 and RTX 50 | RTX 30/20 can run via projects such as `dlssg_for_sm86`, with compatible games/runtime |
| 3x/4x MFG | RTX 50 | RTX 40 NVIDIA-model DLSS-G unlocks are tested; RTX 30 backend/kernel ports exist |
| 5x/6x and Dynamic MFG | RTX 50 | Some unlockers expose these on RTX 30/40, but max multiplier depends on game/plugin/runtime; pacing and compatibility vary |

NVIDIA describes Dynamic MFG as adjusting the multiplier to reach a target frame rate or the display's maximum refresh rate. Its app rollout requires a compatible game/driver; the official dynamic override is not a promise that every community unlocker can use Dynamic mode.

A critical terminology distinction: a mod's x6 output may be **NVIDIA-model DLSS-G MFG** (patched architecture gate plus a compatible NVIDIA runtime), or **hybrid frame generation** (for example, one NVIDIA DLSS-G frame plus FSR 3.1-generated frames on RTX 40, or FSR-generated frames on RTX 20/30). Both can raise the output-frame counter; only the former runs NVIDIA's MFG model for those frames. Check the implementation, not just the menu label.

### 6.4 The workload

NVIDIA says the GPU runs "5 AI models across Super Resolution, Ray Reconstruction, and Multi Frame Generation for each rendered frame, all within a few milliseconds." It also gives a per-frame cost comparison from the ADLR page. DLSS 3 frame generation took about 3.25 ms per 4K frame on an RTX 4090. DLSS 4 MFG took about 1 ms per generated frame on an RTX 5090 at launch. These are different GPUs and different generations, so read them as a trend.

---

## 7. What Blackwell adds beyond the AI cores

![Frame pacing: CPU pacing versus Flip Metering](figures/fig13-flip-metering.svg)

Faster tensor cores explain part of the MFG result. NVIDIA also points to hardware that is not in the SM.

### 7.1 Flip Metering in the display engine

DLSS 3 paced generated frames on the CPU. NVIDIA says that pacing's variability "can compound with additional frames, leading to less consistent frame pacing." Blackwell moves the pacing logic into the **display engine**, which controls the timing of each flip (scanout). NVIDIA calls this hardware Flip Metering.

A third-party project (mfg-unlock) reports that on RTX 40 a CPU pacer inside `sl.dlss_g.dll` replaces the display-engine pacer, and that it produces visible "stacked frames" stutter. That is third-party analysis, not an NVIDIA statement. The open-source PresentMon project (GameTechDev) merged a pull request from an NVIDIA developer that adds driver events for Flip Metering, so measured flip intervals can reflect the display engine's timing. That is also third-party, from the public repository.

### 7.2 Display engine bandwidth

NVIDIA says the Blackwell display engine has "twice the pixel processing capability" to support Flip Metering at high resolution and refresh rate. This is a stated number, not a measured one.

### 7.3 Other blocks NVIDIA lists

NVIDIA's Blackwell architecture material also names:

- **AI Management Processor (AMP):** a scheduler for AI workloads that run alongside graphics. Third-party sources describe it as RISC-V based. I could not confirm that detail from NVIDIA.
- **Neural shader support:** NVIDIA's RTX Neural Shaders, cooperative vectors (a DirectX shader feature that exposes matrix operations to the Tensor cores), and the unified FP32/INT32 datapath.
- **Memory:** GDDR7 with PAM3 signaling.
- **Media:** a 9th-gen NVENC and 6th-gen NVDEC, as listed in the architecture material.

### 7.4 RTX 50 versus RTX 40/30: actual advantages, not just gates

| Capability | RTX 30 (Ampere) | RTX 40 (Ada) | RTX 50 (Blackwell) |
|---|---|---|---|
| FP8 Tensor Core acceleration | No | Yes | Yes, newer Tensor Core generation |
| FP4 Tensor Core acceleration | No | No | Yes |
| Native NVIDIA 2x Frame Generation | Not officially enabled; community SM86/SM75 backends exist | Yes | Yes |
| Official NVIDIA 4x/6x MFG | No | No | Yes |
| Community NVIDIA DLSS-G MFG | SM86 backend/runtime projects; D3D12-focused | Architecture-check patches; tested on Ada | Native |
| Hardware Flip Metering | No | No | Yes, in the display engine |
| DLSS 5 official path | No | RTX 40 expansion planned by NVIDIA, no date | Yes |
| DLSS 5 community path | Patched SM86 PTX / approximate FP16 build demonstrated | Blackwell kernels replaced with Ada-compatible binaries | Native runtime/model path |

**The core architectural gap is efficiency and predictability, not a simple ability/no-ability switch.** The RTX 40 MFG unlock patches the software architecture gate and works around the fact that Ada has no Blackwell display-engine metering. NVIDIA's frame-generation network can execute on Ada Tensor Cores, but scheduling the presents falls back to software/CPU pacing. The KTH thesis found higher frame-time variance on an RTX 4090 than an RTX 5080 even at ordinary 2x FG. A later Tom's Hardware test verified 4x MFG on RTX 40 in *Cyberpunk 2077*: it found the feature visibly smooth on a high-refresh VRR display, but Blackwell scaled better and had lower latency. Those are measured comparisons, not proof every game or card behaves the same.

RTX 30's community routes have more moving parts. The `dlssg_for_sm86` project supplies a compatible DLSS-G runtime/backend and kernel packs for Ampere SM86 (and Turing SM75), currently focusing on Windows/D3D12 games that already integrate DLSS Frame Generation. Its own README reports that some optimized kernels once produced corrupted frames or driver resets after feature recreation; a later release fixed the tested case. It also warns that Vulkan support is experimental/limited. Its published game captures are useful but the author explicitly says they are not a complete frame-time, latency, or long-duration stability study. The separate MFG Unlocker v1.6.1 release notes say its RTX 30 DLSS-G path uses NVIDIA driver's flip-pacing path by default to reduce stalls at 3x and above; this is not the Blackwell display-engine Flip Metering hardware path.

**MFG results are not automatically lower-latency results.** The game still simulates and samples input at the base render rate. A 40-to-160 FPS output can look smoother, but it does not become 160 fresh game-state updates per second. The Tom's Hardware test used about 60 ms as its chosen average-latency threshold; the RTX 4090 through RTX 4070 stayed below it for 4x MFG alone in its test setup, but adding the DLSS 5 mod raised latency past that threshold for most Ada cards at 1440p. This is one workload, one metric, and one monitor/test setup—not a universal promise.

### 7.5 DLSS 5 on RTX 40 and RTX 30

The community DLSS 5 ports target a different stage from MFG: they insert a neural-rendering pass after the game's ordinary rendering. That late position means a wrapper can often feed it color and motion information without rewriting the whole engine integration. It does **not** mean the mod has the developer's full semantic masks, carefully authored motion, HDR path, or title-specific art direction.

- **Ada / RTX 40:** Uncle Burrito's patch replaced Blackwell-only CUDA binaries in the leaked DLSS-NR library with Ada-compatible binaries. A *Control* demo on RTX 4090 was reported around 82 FPS with the pass versus roughly 130–135 FPS without it. Tom's Hardware's combined *Cyberpunk 2077* test found additional latency and wider gaps between average FPS and 1% lows, especially on lower Ada cards. NVIDIA has said official RTX 40 support is planned after further RTX 50 tuning; no date was given.
- **Ampere / RTX 30:** the first FP8-oriented build was extremely slow because these GPUs lack native FP8. Later community builds replaced or recompiled the path for SM86 using FP16. TechPowerUp reported an RTX 2060 community demo and RTX 3080/3090-class cases around 25–35 FPS in selected games/resolutions, while Tom's Hardware collected many single-digit or 20–40 FPS results with noisy/shimmering images. Results vary widely by title/settings; these do not establish near-RTX-50 performance.
- **Why Ampere falls behind:** FP16 is supported, so the model is not fundamentally impossible. But the model/runtime is tuned for FP8 and Blackwell/Ada paths. FP16 uses more compute and memory traffic; Ampere also has fewer Tensor Core generations/throughput and less bandwidth than current high-end Blackwell cards. Community FP16/PTX builds are alternate implementations, not proof of numerical or visual equivalence to the official FP8 runtime.

Treat all old-GPU DLSS 5 demonstrations as **compatibility experiments**, not as supported ports. Some use a leaked proprietary runtime, other builds recompile/approximate kernels, and some all-in-one tools combine it with unrelated frame-generation models. A screenshot or FPS counter alone cannot establish temporal quality, end-to-end latency, stability, or that the intended NVIDIA model generated every displayed frame.

---

## 8. Model architectures

![How the DLSS models changed architecture](figures/fig14-model-architectures.svg)

### 8.1 Super Resolution and Ray Reconstruction

**DLSS 2 and 3 (CNN).** A temporal convolutional network. Inputs are jittered low-resolution color, motion vectors, depth, and exposure. "Presets" A to F are pretrained variants of the same network, chosen per quality mode. NVIDIA later removed some of them and deprecated others, and says the CNN "reached the limits of what's possible." NVIDIA has not published the layer structure.

**DLSS 4 (vision Transformer).** NVIDIA says the model uses self-attention "to evaluate the relative importance of each pixel across the entire frame, and over multiple frames," with "double the parameters" of the CNN. NVIDIA's ADLR page says the transformer is written for NVIDIA's GPU architecture, with custom CUDA kernels that use on-chip memory and FP8. A third-party analysis of the Preset K kernel describes a U-shaped network with 8x8 windowed attention. NVIDIA has not confirmed that, so treat it as inference.

**DLSS 4.5 (second-generation Transformer).** NVIDIA says it uses five times the compute of the first-generation Transformer, and that it was trained on a larger dataset. It uses the engine's pixel sampling and motion data "more intelligently." It has presets M (Performance), L (Ultra Performance at 4K), and K (other modes). NVIDIA does not publish a parameter count.

**Ray Reconstruction.** RR takes the noisy ray-traced output along with the same G-buffer data (depth, normals, motion vectors) and fuses denoising with upscaling in one network. Preset F is the 2nd-gen RR model. NVIDIA says it has 35% more compute capability and 20% more parameters than the previous model, at similar performance.

### 8.2 Frame Generation

**DLSS 3 FG (CNN autoencoder).** Four inputs: the current frame, the prior frame, the OFA flow field, and engine motion vectors and depth. The output is one generated frame. The hudless color and UI buffers are passed in separately so the network can keep the HUD out of its output.

**DLSS 4 FG and MFG (learned flow plus split network).** The OFA is replaced by a learned optical-flow model. The synthesis network is split, as described in section 6. The new model runs on RTX 40 as well as RTX 50.

**DLSS 4.5 (Preset B).** A Frame Generation model that reads extra UI buffers from selected engines. Tom's Hardware reports that it runs on RTX 40 and RTX 50. I could not confirm that on NVIDIA's own pages, so treat it as reported.

### 8.3 DLSS 5

DLSS 5 is described in NVIDIA's ADLR paper as a one-step pixel-space diffusion model. Its inputs at inference are the rendered color frame, engine motion vectors, carried temporal state, and artistic-direction values. During training it also sees renderer-derived scene attributes that supervise the output to stay faithful to the authored scene. NVIDIA says it is deterministic and causal, and that it runs under a fixed per-frame compute budget.

NVIDIA's October 2026 update says the model is now about 118 million parameters, 24% smaller than before, with half the GPU overhead of the September version. Other reports put the earlier model at 148 to 155 million parameters, but those figures come from press calculations, not from NVIDIA.

---

## 9. Risks and boundaries of community builds

- **Per-game compatibility:** most native MFG unlocks need a game with Streamline DLSS Frame Generation already wired in. API, Streamline/plugin version, bundled `nvngx_dlssg.dll`, and per-game multiplier caps can determine whether a mode initializes. Some older games need a newer runtime; changing binaries can also break the game.
- **Pacing is a feature, not a checkbox:** unlock only, without a pacing workaround, can produce stacked or uneven frames. Ada/Ampere do not gain Blackwell's hardware Flip Metering when a software check is patched. Community pacers may improve delivery but are not the same hardware path.
- **Generated FPS is not base FPS:** interpolation does not run game simulation or sample new input for each generated frame. High multipliers make a weak base frame rate more visible and can add artifacts/latency. Use frame-time, presented/displayed-time, and input-latency measurements—not just an overlay FPS number.
- **Visual quality and buffer inputs:** motion-vector errors, disocclusion, particle effects, reflections, HUD composition, or a synthetic motion field can create ghosting, flicker, UI junk, and shimmer. A wrapper may not receive the same buffers as a native game integration.
- **Anti-cheat and stability:** proxy DLLs/add-ons modify or hook a game's render process. Do not use them in multiplayer or anti-cheat-protected sessions unless the game publisher explicitly permits them. They can trigger anti-cheat, crash after a driver/game update, conflict with overlays, or corrupt output.
- **DLSS 5 redistribution/security:** many DLSS 5 experiments use leaked/proprietary runtime files that projects ask users to supply themselves. Avoid unofficial download mirrors and installers; use the project's source/release page, verify files, and understand the licence/redistribution uncertainty.

These are community experiments. Verify the current release notes and game compatibility before testing, keep a clean copy of game files, and use single-player/offline titles.

---

## 10. What NVIDIA does not publish

- Layer counts, token sizes, and attention layouts for the SR and RR transformers.
- Whether the OFA block exists on RTX 50 silicon.
- The exact timestep conditioning used by MFG.
- The input color space DLSS expects before tonemapping (check the Streamline programming guide before writing engine code).
- Measured pacing results with Flip Metering on and off across all games.
- A guarantee of visual/latency parity for the unofficial Ada/Ampere ports.

Where a claim depends on one of these, this document says so.

---

## 11. Sources

NVIDIA:

- Ampere GA102 whitepaper: https://www.nvidia.com/content/PDF/nvidia-ampere-ga-102-gpu-architecture-whitepaper-v2.pdf
- Ada Lovelace whitepaper: https://images.nvidia.com/aem-dam/Solutions/geforce/ada/nvidia-ada-gpu-architecture.pdf
- RTX Blackwell GPU architecture whitepaper: https://images.nvidia.com/aem-dam/Solutions/geforce/blackwell/nvidia-rtx-blackwell-gpu-architecture.pdf
- RTX 4090 and RTX 5090 spec pages: https://www.nvidia.com/en-us/geforce/graphics-cards/40-series/rtx-4090/ and https://www.nvidia.com/en-us/geforce/graphics-cards/50-series/rtx-5090/
- Optical Flow SDK programming guide: https://docs.nvidia.com/video-technologies/optical-flow-sdk/nvofa-programming-guide/index.html
- DLSS 3 announcement (Sept 2022): https://www.nvidia.com/en-us/geforce/news/dlss3-ai-powered-neural-graphics-innovations/
- DLSS 4 Multi Frame Generation (6 Jan 2025): https://www.nvidia.com/en-us/geforce/news/dlss4-multi-frame-generation-ai-innovations/
- ADLR DLSS 4: https://research.nvidia.com/labs/adlr/DLSS4/
- ADLR DLSS 5 (1 Sep 2026): https://research.nvidia.com/labs/adlr/DLSS5/
- DLSS 4.5 Super Resolution (14 Jan 2026): https://www.nvidia.com/en-us/geforce/news/dlss-4-5-super-resolution-available-now/
- DLSS 4.5 Ray Reconstruction (25 Aug 2026): https://www.nvidia.com/en-us/geforce/news/gamescom-2026-dlss-4-5-ray-reconstruction-release-announcements-trailers/
- DLSS 4.5 Dynamic MFG and 6x (31 Mar 2026): https://www.nvidia.com/en-us/geforce/news/dlss-4-5-dynamic-multi-frame-generation-6x-mode-released/
- DLSS 5 October update (13 Oct 2026): https://www.nvidia.com/en-us/geforce/news/dlss-5-faster-ai-model-hogwarts-legacy-rtx-spark-laptops/

Third-party, used only where noted:

- Digital Foundry, Bryan Catanzaro interview (CES 2025): https://www.digitalfoundry.net/articles/digitalfoundry-2025-bryan-catanzaro-interview-dlss-4-and-machine-learning
- mfg-unlock project (reverse engineering of `nvngx_dlssg.dll`): https://deepwiki.com/matiasLombo/mfg-unlock/3.2-multi-frame-generation-pipeline
- PresentMon pull request adding DLSS 4 flip-metering events: https://github.com/GameTechDev/PresentMon/pull/440
- Tom's Hardware, DLSS 4.5 Dynamic MFG and Preset B: https://www.tomshardware.com/pc-components/gpus/nvidia-dlss-4-5-dynamic-mfg-plus-5x-and-6x-framegen-modes-enter-beta-for-rtx-50-series-users-update-offers-greater-control-over-generated-frame-rates-and-more-headroom-for-high-refresh-rate-displays
- Tom's Hardware, tested MFG on RTX 40: https://www.tomshardware.com/pc-components/gpus/we-tested-dlss-multi-frame-generation-on-rtx-40-series-gpus-new-mod-brings-rtx-50-series-exclusive-feature-to-older-cards-and-it-really-works
- Tom's Hardware, interviews with the MFG and DLSS 5 modders: https://www.tomshardware.com/pc-components/gpus/meet-the-modders-porting-ringfenced-dlss-features-to-older-gpus-how-dlss-5-and-multi-frame-generation-were-unlocked-for-older-architectures
- MFG Unlocker / RTX MFG Universal, community documentation and releases: https://github.com/dashdogy/MFG-Unlocker
- `mfg-unlock`, Ada gate and pacing patches: https://github.com/matiasLombo/mfg-unlock
- `dlssg_for_sm86`, Ampere/Turing compatible backend and runtime: https://github.com/sdli1995/dlssg_for_sm86
- `dlss-unlocked`, hybrid route bundler: https://github.com/ShyVortex/dlss-unlocked
- TechPowerUp, DLSS 5 on older GPUs / FP16 builds: https://www.techpowerup.com/review/nvidia-dlss-5-technical-preview/5.html
- Tom's Hardware, DLSS 5 on Ampere community tests: https://www.tomshardware.com/pc-components/gpus/dlss-5-mod-brings-next-gen-tech-to-old-ampere-gpus-but-frame-rates-are-horrible-most-games-tank-to-single-digits-high-end-gpus-can-hit-up-to-40-fps-in-some-cases
- KTH thesis, MFG feasibility on RTX 4090: https://www.diva-portal.org/smash/get/diva2:1985716/FULLTEXT01.pdf
