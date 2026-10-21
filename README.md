# How NVIDIA DLSS Works: Models, Architecture, and Game Integration

A blog post, with figures, explaining the NVIDIA DLSS family of neural models:
Super Resolution, Ray Reconstruction, Frame Generation, Multi Frame Generation,
and DLSS 5 neural rendering. It also covers how games call these models through
NVIDIA Streamline, and why community injectors such as ReShade and OptiScaler
can break them.

## Repository layout

```
.
├── post.md                  # The blog post (Markdown, references figures/)
├── figures/                 # Original SVG diagrams (hand-written, no external art)
│   ├── fig1-pipeline.svg
│   ├── fig2-jitter.svg
│   ├── fig3-motion-vectors.svg
│   ├── fig4-frame-generation.svg
│   ├── fig5-hudless.svg
│   └── fig6-injector-stack.svg
├── scripts/
│   └── render-figures.ps1   # Exports SVGs to PNG for platforms that need raster images
├── .gitignore
└── README.md
```

## Reading the post

Open `post.md` in any Markdown viewer. Images use relative paths, so keep the
`figures/` folder next to the post.

## Sources and fact checking

Most technical claims in `post.md` are taken from NVIDIA's own announcements or
developer documentation. The exceptions are marked in the post. Key sources:

- NVIDIA GeForce news: DLSS 4 (CES 2025), DLSS 4.5 (CES 2026), DLSS 3.5 Ray Reconstruction (2023), DLSS 5 (GTC, March 2026)
- NVIDIA Streamline SDK: https://github.com/NVIDIA-RTX/Streamline
- NVIDIA DLSS Programming Guide: https://github.com/NVIDIA/DLSS
- NVIDIA developer blog, DLSS 4.5 Super Resolution and Dynamic Multi Frame Generation: https://developer.nvidia.com/blog/nvidia-dlss-4-5-delivers-super-resolution-upgrades-and-new-dynamic-multi-frame-generation/

Items still tagged `[VERIFY]` in `post.md` need a check before publishing. Do not
publish claims marked that way.

The DLSS 5 section describes official NVIDIA positioning only. Running neural
rendering on RTX 40 hardware is unofficial and is described as such.

## Render PNG figures

Many blog platforms, including Medium, do not accept SVG. Export PNGs with one
of these tools (install one):

| Tool | Install (Windows) |
|---|---|
| rsvg-convert | MSYS2: `pacman -S mingw-w64-x86_64-librsvg` (puts `rsvg-convert` on PATH) |
| Inkscape | `winget install --id Inkscape.Inkscape` |
| ImageMagick | `winget install --id ImageMagick.ImageMagick` |

Then run, from the repo root:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\render-figures.ps1
```

The PNGs are written to `figures/png/`, which is excluded from git.

## Publishing

1. Render the PNGs (above).
2. Paste the contents of `post.md` into your platform.
3. Upload each figure from `figures/png/` at its position in the post. The
   alt text is the caption under each image in `post.md`.
4. Replace the `[VERIFY]` items, or remove them, before publishing.

## Working on the post

- Edit `post.md` directly. Keep figure paths relative.
- Edit SVG figures with any text editor. They are plain XML. Re-run
  `scripts/render-figures.ps1` after changing them.
- Commit changes with a message that says what changed in the post, for example
  `post: correct DLSS 4.5 preset mapping`.

## License

The text and SVG figures are original. Add a license here before publishing if
you want to control reuse. Do not add NVIDIA screenshots or logos without
permission.
