# Generates animated looping GIFs in figures/ using built-in Windows .NET System.Drawing + WPF GifBitmapEncoder.
# Run from repo root:
#   powershell -ExecutionPolicy Bypass -File scripts\generate-gifs.ps1

Add-Type -AssemblyName System.Drawing, PresentationCore, WindowsBase

$root = Split-Path -Parent $PSScriptRoot
$figDir = Join-Path $root "figures"
New-Item -ItemType Directory -Force -Path $figDir | Out-Null

function Save-AnimatedGif {
    param(
        [string]$OutputPath,
        [System.Drawing.Bitmap[]]$Frames,
        [int]$DelayCs = 80 # centiseconds per frame (80 = 0.8s)
    )
    # Encode each frame with GifBitmapEncoder to extract global color table / image descriptor,
    # or write a clean GIF89a stream with NETSCAPE2.0 loop extension and Graphic Control Extension per frame.
    $enc = New-Object System.Windows.Media.Imaging.GifBitmapEncoder
    foreach ($bmp in $Frames) {
        $hBitmap = $bmp.GetHbitmap()
        try {
            $src = [System.Windows.Interop.Imaging]::CreateBitmapSourceFromHBitmap(
                $hBitmap, [IntPtr]::Zero, [System.Windows.Int32Rect]::Empty,
                [System.Windows.Media.Imaging.BitmapSizeOptions]::FromEmptyOptions()
            )
            $enc.Frames.Add([System.Windows.Media.Imaging.BitmapFrame]::Create($src))
        } finally {
            # Clean up GDI handle
            Add-Type -MemberDefinition '[DllImport("gdi32.dll")] public static extern bool DeleteObject(IntPtr hObject);' -Name Gdi -Namespace Win32 -ErrorAction SilentlyContinue
            [Win32.Gdi]::DeleteObject($hBitmap) | Out-Null
        }
    }

    $ms = New-Object System.IO.MemoryStream
    $enc.Save($ms)
    $raw = $ms.ToArray()
    $ms.Dispose()

    # Inject NETSCAPE2.0 looping extension and per-frame delay into GIF89a byte stream
    $out = New-Object System.IO.MemoryStream
    # Header (6 bytes) + Logical Screen Descriptor (7 bytes)
    $headerLen = 13
    if (($raw[10] -band 0x80) -ne 0) {
        $gctSize = 3 * [Math]::Pow(2, (($raw[10] -band 0x07) + 1))
        $headerLen += [int]$gctSize
    }
    $out.Write($raw, 0, $headerLen)

    # NETSCAPE2.0 infinite loop block
    [byte[]]$loopExt = @(0x21, 0xFF, 0x0B, 0x4E, 0x45, 0x54, 0x53, 0x43, 0x41, 0x50, 0x45, 0x32, 0x2E, 0x30, 0x03, 0x01, 0x00, 0x00, 0x00)
    $out.Write($loopExt, 0, $loopExt.Length)

    $lo = [byte]($DelayCs -band 0xFF)
    $hi = [byte](($DelayCs -shr 8) -band 0xFF)

    $i = $headerLen
    while ($i -lt $raw.Length) {
        if ($raw[$i] -eq 0x21 -and $raw[$i + 1] -eq 0xF9) {
            # Existing Graphic Control Extension (8 bytes): overwrite delay bytes at +4, +5
            $raw[$i + 4] = $lo
            $raw[$i + 5] = $hi
            $out.Write($raw, $i, 8)
            $i += 8
        } elseif ($raw[$i] -eq 0x2C) {
            # Image Descriptor without preceding GCE -> insert GCE before image descriptor
            [byte[]]$gce = @(0x21, 0xF9, 0x04, 0x04, $lo, $hi, 0x00, 0x00)
            $out.Write($gce, 0, $gce.Length)
            # Copy image descriptor (10 bytes) + local color table if present + LZW data blocks
            $hasLct = ($raw[$i + 9] -band 0x80) -ne 0
            $descLen = 10
            if ($hasLct) {
                $descLen += [int](3 * [Math]::Pow(2, (($raw[$i + 9] -band 0x07) + 1)))
            }
            $out.Write($raw, $i, $descLen)
            $i += $descLen
            # LZW minimum code size byte
            $out.WriteByte($raw[$i])
            $i++
            # Sub-blocks
            while ($i -lt $raw.Length -and $raw[$i] -ne 0) {
                $blockSize = [int]$raw[$i]
                $out.Write($raw, $i, $blockSize + 1)
                $i += $blockSize + 1
            }
            if ($i -lt $raw.Length) {
                $out.WriteByte(0)
                $i++
            }
        } else {
            $out.WriteByte($raw[$i])
            $i++
        }
    }

    [System.IO.File]::WriteAllBytes($OutputPath, $out.ToArray())
    $out.Dispose()
    foreach ($bmp in $Frames) { $bmp.Dispose() }
    Write-Host "Generated $OutputPath"
}

function New-Canvas([int]$w, [int]$h) {
    $bmp = New-Object System.Drawing.Bitmap($w, $h)
    $g = [System.Drawing.Graphics]::FromImage($bmp)
    $g.SmoothingMode = [System.Drawing.Drawing2D.SmoothingMode]::AntiAlias
    $g.TextRenderingHint = [System.Drawing.Text.TextRenderingHint]::ClearTypeGridFit
    $g.Clear([System.Drawing.Color]::White)
    return @{ Bmp = $bmp; G = $g }
}

$fontTitle = New-Object System.Drawing.Font("Segoe UI", 14, [System.Drawing.FontStyle]::Bold)
$fontSub   = New-Object System.Drawing.Font("Segoe UI", 10.5, [System.Drawing.FontStyle]::Regular)
$fontBold  = New-Object System.Drawing.Font("Segoe UI", 10.5, [System.Drawing.FontStyle]::Bold)
$fontSmall = New-Object System.Drawing.Font("Segoe UI", 9.5, [System.Drawing.FontStyle]::Regular)

$brDark   = New-Object System.Drawing.SolidBrush([System.Drawing.Color]::FromArgb(26, 26, 26))
$brMuted  = New-Object System.Drawing.SolidBrush([System.Drawing.Color]::FromArgb(80, 80, 80))
$brBlue   = New-Object System.Drawing.SolidBrush([System.Drawing.Color]::FromArgb(58, 107, 196))
$brRed    = New-Object System.Drawing.SolidBrush([System.Drawing.Color]::FromArgb(192, 57, 43))
$brGreen  = New-Object System.Drawing.SolidBrush([System.Drawing.Color]::FromArgb(118, 185, 0))
$brOrange = New-Object System.Drawing.SolidBrush([System.Drawing.Color]::FromArgb(199, 119, 0))

# ==============================================================================
# GIF 1: Sub-pixel Jitter & Temporal Accumulation (anim-jitter-accumulation.gif)
# ==============================================================================
$jitterFrames = @()
$offsets = @(
    @{ dx = -0.25; dy = -0.25; Brush = $brBlue;   Name = "Frame 1: Jitter (-0.25, -0.25)" },
    @{ dx =  0.25; dy =  0.25; Brush = $brRed;    Name = "Frame 2: Jitter (+0.25, +0.25)" },
    @{ dx =  0.25; dy = -0.25; Brush = $brGreen;  Name = "Frame 3: Jitter (+0.25, -0.25)" },
    @{ dx = -0.25; dy =  0.25; Brush = $brOrange; Name = "Frame 4: Jitter (-0.25, +0.25) -> Full 4x Density" }
)

for ($step = 0; $step -lt 4; $step++) {
    $c = New-Canvas 720 300
    $g = $c.G
    $g.DrawString("DLSS Super Resolution: Sub-Pixel Jitter Accumulation", $fontTitle, $brDark, 24, 16)
    $g.DrawString($offsets[$step].Name, $fontBold, $offsets[$step].Brush, 24, 46)

    # Draw 4x4 low-res pixel cells ( subdivided into 2x2 target grid )
    $gx = 40; $gy = 80; $cell = 46
    $penSub = New-Object System.Drawing.Pen([System.Drawing.Color]::FromArgb(220, 220, 220), 1)
    $penMaj = New-Object System.Drawing.Pen([System.Drawing.Color]::FromArgb(120, 120, 120), 2)
    for ($i = 0; $i -le 8; $i++) {
        $p = if ($i % 2 -eq 0) { $penMaj } else { $penSub }
        $g.DrawLine($p, $gx + $i * ($cell / 2), $gy, $gx + $i * ($cell / 2), $gy + 4 * $cell)
        $g.DrawLine($p, $gx, $gy + $i * ($cell / 2), $gx + 4 * $cell, $gy + $i * ($cell / 2))
    }

    for ($s = 0; $s -le $step; $s++) {
        $ox = $offsets[$s].dx; $oy = $offsets[$s].dy
        $b  = $offsets[$s].Brush
        for ($row = 0; $row -lt 4; $row++) {
            for ($col = 0; $col -lt 4; $col++) {
                $cx = $gx + ($col + 0.5 + $ox) * $cell
                $cy = $gy + ($row + 0.5 + $oy) * $cell
                $r = if ($s -eq $step) { 6 } else { 4 }
                $g.FillEllipse($b, [float]($cx - $r), [float]($cy - $r), [float](2 * $r), [float](2 * $r))
            }
        }
    }

    # Right-side explanation panel
    $rx = 265
    $g.DrawString("How Temporal History Reconstructs 4K:", $fontBold, $brDark, $rx, 82)
    $g.DrawString("1. Engine renders at lower resolution (bold grid)", $fontSmall, $brMuted, $rx, 110)
    $g.DrawString("2. Camera shifts by a Halton sub-pixel offset each frame", $fontSmall, $brMuted, $rx, 134)
    $g.DrawString("3. Motion vectors align prior samples to current frame", $fontSmall, $brMuted, $rx, 158)
    $g.DrawString("4. Transformer self-attention weights valid history", $fontSmall, $brMuted, $rx, 182)
    $samplesCount = ($step + 1) * 16
    $g.DrawString("Accumulated sub-pixel samples: $samplesCount / 64", $fontBold, $brDark, $rx, 218)

    # Progress bar
    $g.FillRectangle((New-Object System.Drawing.SolidBrush([System.Drawing.Color]::FromArgb(235, 235, 235))), $rx, 244, 380, 18)
    $g.FillRectangle($brGreen, $rx, 244, [int](380 * ($step + 1) / 4.0), 18)

    $g.Dispose()
    $jitterFrames += $c.Bmp
}
Save-AnimatedGif -OutputPath (Join-Path $figDir "anim-jitter-accumulation.gif") -Frames $jitterFrames -DelayCs 95

# ==============================================================================
# GIF 2: Multi Frame Generation Pacing Timeline (anim-mfg-pacing.gif)
# ==============================================================================
$mfgFrames = @()
for ($tick = 0; $tick -lt 6; $tick++) {
    $c = New-Canvas 780 315
    $g = $c.G
    $g.DrawString("DLSS Frame Generation & Multi Frame Generation (2x / 4x / 6x)", $fontTitle, $brDark, 24, 16)
    $g.DrawString("Display scanout step $($tick + 1) of 6 between two rendered engine frames (Hardware Flip Metering)", $fontSub, $brMuted, 24, 44)

    $modes = @(
        @{ Label = "Native (1x)";       Slots = @(1,0,0,0,0,0) },
        @{ Label = "DLSS FG (2x)";      Slots = @(1,0,0,2,0,0) },
        @{ Label = "DLSS 4 MFG (4x)";   Slots = @(1,2,2,2,0,0) },
        @{ Label = "DLSS 4.5 MFG (6x)"; Slots = @(1,2,2,2,2,2) }
    )

    for ($m = 0; $m -lt $modes.Count; $m++) {
        $y = 84 + $m * 50
        $g.DrawString($modes[$m].Label, $fontBold, $brDark, 24, $y + 8)
        for ($s = 0; $s -lt 6; $s++) {
            $x = 215 + $s * 86
            $type = $modes[$m].Slots[$s]
            $active = ($s -le $tick)
            if ($type -eq 1) {
                $fill = if ($active) { $brBlue } else { New-Object System.Drawing.SolidBrush([System.Drawing.Color]::FromArgb(220, 230, 245)) }
                $g.FillRectangle($fill, $x, $y, 74, 34)
                $g.DrawString("Real", $fontSmall, [System.Drawing.Brushes]::White, $x + 21, $y + 8)
            } elseif ($type -eq 2) {
                $fill = if ($active) { $brGreen } else { New-Object System.Drawing.SolidBrush([System.Drawing.Color]::FromArgb(230, 242, 215)) }
                $g.FillRectangle($fill, $x, $y, 74, 34)
                $g.DrawString("AI Gen", $fontSmall, $brDark, $x + 15, $y + 8)
            } else {
                $penDash = New-Object System.Drawing.Pen([System.Drawing.Color]::FromArgb(200, 200, 200), 1)
                $g.DrawRectangle($penDash, $x, $y, 74, 34)
            }
            if ($s -eq $tick) {
                $penHi = New-Object System.Drawing.Pen([System.Drawing.Color]::FromArgb(26, 26, 26), 2.5)
                $g.DrawRectangle($penHi, $x - 2, $y - 2, 78, 38)
            }
        }
    }
    $g.DrawString("Blue = Engine Rendered Frame | Green = AI Generated Frame (Hudless Scene + Recomposited UI)", $fontSmall, $brMuted, 24, 284)
    $g.Dispose()
    $mfgFrames += $c.Bmp
}
Save-AnimatedGif -OutputPath (Join-Path $figDir "anim-mfg-pacing.gif") -Frames $mfgFrames -DelayCs 70

# ==============================================================================
# GIF 3: NVIDIA Reflex 2 Frame Warp & Predictive Inpainting (anim-reflex2-frame-warp.gif)
# ==============================================================================
$reflexFrames = @()
$stages = @(
    @{ Step = 1; Title = "Stage 1: GPU Renders Frame N at Camera Pose t0"; Desc = "Reflex 1 synchronizes CPU submission, eliminating the GPU render queue (~50% latency cut)." },
    @{ Step = 2; Title = "Stage 2: Player Moves Mouse Right Before Display Scanout (t1)"; Desc = "While GPU finishes Frame N, CPU samples the newest mouse input and computes updated camera t1." },
    @{ Step = 3; Title = "Stage 3: Frame Warp Shifts Rendered Image to Camera t1"; Desc = "Late reprojection aligns aim immediately, but exposes disoccluded pixel gaps along edges (red)." },
    @{ Step = 4; Title = "Stage 4: Predictive Inpainting Fills Disocclusion Holes"; Desc = "Prior-frame camera, color, and depth fill missing pixels -> up to 75% total latency reduction." }
)

foreach ($st in $stages) {
    $c = New-Canvas 780 300
    $g = $c.G
    $g.DrawString("Reflex 2 Frame Warp (announced, not yet shipping)", $fontTitle, $brDark, 24, 16)
    $g.DrawString($st.Title, $fontBold, $brBlue, 24, 46)
    $g.DrawString($st.Desc, $fontSmall, $brMuted, 24, 68)

    # Viewport box
    $vx = 40; $vy = 100; $vw = 320; $vh = 165
    $g.FillRectangle((New-Object System.Drawing.SolidBrush([System.Drawing.Color]::FromArgb(236, 243, 255))), $vx, $vy, $vw, $vh)
    $g.DrawRectangle((New-Object System.Drawing.Pen([System.Drawing.Color]::FromArgb(80, 80, 80), 2)), $vx, $vy, $vw, $vh)

    if ($st.Step -eq 1) {
        $g.FillEllipse($brBlue, $vx + 140, $vy + 62, 42, 42)
        $g.DrawLine([System.Drawing.Pens]::Black, $vx + 161, $vy + 45, $vx + 161, $vy + 120)
        $g.DrawLine([System.Drawing.Pens]::Black, $vx + 125, $vy + 83, $vx + 197, $vy + 83)
    } elseif ($st.Step -eq 2) {
        $g.FillEllipse($brBlue, $vx + 140, $vy + 62, 42, 42)
        $g.FillEllipse($brOrange, $vx + 200, $vy + 52, 42, 42)
        $g.DrawLine((New-Object System.Drawing.Pen([System.Drawing.Color]::FromArgb(199, 119, 0), 2.5)), $vx + 161, $vy + 83, $vx + 221, $vy + 73)
    } elseif ($st.Step -eq 3) {
        # Shifted scene left, exposing red disocclusion strip on right and behind target
        $g.FillRectangle($brRed, $vx + $vw - 36, $vy + 2, 34, $vh - 4)
        $g.FillEllipse($brRed, $vx + 118, $vy + 62, 44, 44)
        $g.FillEllipse($brBlue, $vx + 96, $vy + 68, 42, 42)
    } else {
        # Inpainted clean frame
        $g.FillRectangle((New-Object System.Drawing.SolidBrush([System.Drawing.Color]::FromArgb(215, 240, 190))), $vx + $vw - 36, $vy + 2, 34, $vh - 4)
        $g.FillEllipse($brBlue, $vx + 96, $vy + 68, 42, 42)
        $g.DrawLine([System.Drawing.Pens]::Black, $vx + 161, $vy + 45, $vx + 161, $vy + 120)
        $g.DrawLine([System.Drawing.Pens]::Black, $vx + 125, $vy + 83, $vx + 197, $vy + 83)
    }

    # Right side latency summary
    $lx = 390
    $g.DrawString("Latency Breakdown (THE FINALS, 4K Max):", $fontBold, $brDark, $lx, 105)
    $g.FillRectangle($brRed,    $lx, 138, 180, 22); $g.DrawString("Native: 56 ms", $fontSmall, $brDark, $lx + 188, 140)
    $g.FillRectangle($brOrange, $lx, 172, 88,  22); $g.DrawString("Reflex 1 (Low Latency): 27 ms (-50%)", $fontSmall, $brDark, $lx + 96, 174)
    $g.FillRectangle($brGreen,  $lx, 206, 46,  22); $g.DrawString("Reflex 2 (Frame Warp): 14 ms (-75%)", $fontBold, $brDark, $lx + 54, 208)

    $g.Dispose()
    $reflexFrames += $c.Bmp
}
Save-AnimatedGif -OutputPath (Join-Path $figDir "anim-reflex2-frame-warp.gif") -Frames $reflexFrames -DelayCs 120
