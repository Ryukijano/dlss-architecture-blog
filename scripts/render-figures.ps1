# Renders figures/*.svg to figures/png/*.png for blog platforms that do not accept SVG.
# Uses the first converter found: rsvg-convert, Inkscape, or ImageMagick.
# Install one of them first (see README.md, "Render PNG figures").

$root = Split-Path -Parent $PSScriptRoot
$src  = Join-Path $root "figures"
$out  = Join-Path $src "png"
New-Item -ItemType Directory -Force -Path $out | Out-Null

$rsvg     = Get-Command rsvg-convert -ErrorAction SilentlyContinue
$inkscape = Get-Command inkscape -ErrorAction SilentlyContinue
$magick   = Get-Command magick -ErrorAction SilentlyContinue

if (-not ($rsvg -or $inkscape -or $magick)) {
    Write-Error "No SVG converter found. Install rsvg-convert, Inkscape, or ImageMagick (see README.md)."
    exit 1
}

Get-ChildItem $src -Filter *.svg | ForEach-Object {
    $dst = Join-Path $out ($_.BaseName + ".png")
    if ($rsvg) {
        & $rsvg.Source -z 2 -o $dst $_.FullName
    } elseif ($inkscape) {
        & $inkscape.Source $_.FullName --export-type=png --export-filename=$dst --export-dpi=192 | Out-Null
    } else {
        & $magick.Source -density 192 $_.FullName $dst
    }
    Write-Host "rendered $($_.Name) -> $dst"
}
