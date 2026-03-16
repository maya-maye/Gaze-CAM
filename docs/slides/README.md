# Presentation Slides

## Building the PDF

```bash
cd docs/slides
pdflatex slides.tex
pdflatex slides.tex   # run twice for references
```

Requires: `texlive-latex-extra` (for Beamer), `texlive-fonts-recommended`, and the `metropolis` theme (`texlive-latex-extra` usually includes it).

Install on Ubuntu/Debian:
```bash
sudo apt install texlive-latex-extra texlive-fonts-recommended texlive-fonts-extra
```

Or via Docker:
```bash
docker run --rm -v "$PWD":/docs -w /docs texlive/texlive pdflatex slides.tex
```

## Animated GIFs in the Presentation

PDF viewers don't natively support GIF playback. Options:

1. **Adobe Acrobat (best):** The `\movie` command on slide 2 creates a clickable region. Click the teaser image in Acrobat to play the GIF.

2. **Live demo during talk:** Open the GIFs from `gifs/` directly in a browser or image viewer alongside the PDF. Switch to them when you hit the qualitative slides.

3. **Convert GIFs to MP4 and embed:**
   ```bash
   for f in gifs/*.gif; do
       ffmpeg -i "$f" -movflags +faststart -pix_fmt yuv420p "${f%.gif}.mp4"
   done
   ```
   Then replace `\movie{...}{gifs/foo.gif}` with `\movie{...}{gifs/foo.mp4}`.

4. **PowerPoint/Google Slides:** If GIF support matters most, export the Beamer slides as images and import into Slides/PPTX, then drag the GIFs directly onto slides — they'll animate natively.

## Slide Layout (20 slides)

| # | Title | Visual |
|---|-------|--------|
| 1 | Title | — |
| 2 | Motivation | GIF (clickable) |
| 3 | Teaser figure | `fig_teaser.png` |
| 4 | Dataset: EGTEA Gaze+ | — |
| 5 | Four Architectures | Table |
| 6 | Pipeline | Text diagram |
| 7 | Evaluation Metrics | — |
| 8 | Classification Accuracy | Table |
| 9 | Gaze–Saliency Alignment | Table |
| 10 | Accuracy vs. Alignment | `fig_comparison_split1.png` |
| 11 | Qualitative Example | `sample_strip.png` / GIF |
| 12 | Correct vs. Incorrect | `fig_correct_vs_incorrect.png` |
| 13 | Error Buckets | `fig_error_buckets.png` |
| 14 | Center-Bias Baseline | Table |
| 15 | Occlusion Sensitivity | Table |
| 16 | Model Randomization | Table |
| 17 | Temporal Lead/Lag | `fig_temporal_lag.png` |
| 18 | Per-Verb Alignment | `fig_verb_alignment.png` |
| 19 | Key Takeaways | — |
| 20 | Limitations & Questions | — |
