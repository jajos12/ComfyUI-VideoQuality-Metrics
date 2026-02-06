# ComfyUI Video Quality Metrics

A comprehensive suite of video quality assessment metrics for ComfyUI.

## Features

### Fidelity Metrics
- **PSNR** - Peak Signal-to-Noise Ratio
- **SSIM** - Structural Similarity Index
- **CIEDE2000** - Perceptually uniform color accuracy

### Temporal Metrics
- **Warping Error** - Optical flow-based temporal consistency
- **Flickering Detection** - Brightness variance analysis
- **Motion Smoothness** - Jerk-based motion quality

### Distributional Metrics
- **FVD** - Fréchet Video Distance
- **FID** - Fréchet Inception Distance

### No-Reference Metrics (Deep Learning)
- **CLIP-IQA** - Aesthetic Scoring & Text Alignment
- **DOVER** - Disentangled Aesthetic & Technical Quality
- **FAST-VQA** - Efficient High-Res Quality Assessment

### Visualization & Reporting
- **Radar Charts** - Multi-metric comparison
- **JSON Export** - Metrics logging
- **Statistical Tests** - T-test, Wilcoxon for workflow comparison

## Installation

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/yourusername/ComfyUI-VideoQuality-Metrics.git
pip install -r ComfyUI-VideoQuality-Metrics/requirements.txt
```

Restart ComfyUI. Nodes appear under `VideoQuality/` category.

## Video & Batch Handling

All nodes accept **Video Tensors** `[T, H, W, 3]` and **Image Batches** `[N, H, W, 3]`.

- **Fidelity Metrics** (PSNR, SSIM, ΔE2000): Return the **average** quality across all frames in the batch.
- **Temporal Metrics**: Require at least 2 frames (T ≥ 2) to compute motion consistency.
- **Distributional Metrics**: Compare the entire distribution of the generated batch against the reference batch.

## Node Reference

| Node | Inputs | Outputs |
|------|--------|---------|
| VQ Full-Reference Metrics | images, reference | psnr, ssim, ciede2000, summary |
| VQ Temporal Consistency | video_frames | warping_error, flickering_score, summary |
| VQ Motion Smoothness | video_frames | smoothness_score, mean_jerk, summary |
| VQ FVD | video_generated, video_reference | fvd, summary |
| VQ FID | images_generated, images_reference | fid, summary |
| VQ CLIP Aesthetic | images | aesthetic_score, summary |
| VQ Text Alignment | video, prompt | alignment_score, summary |
| VQ DOVER Quality | video | aesthetic, technical, overall, summary |
| VQ FAST-VQA | video | quality_score, summary |
| VQ Radar Chart | psnr, ssim, ... | radar_chart, metrics_json |
| VQ Metrics Logger | metrics | json_output |

## Interpretation Guidelines

| Metric | Excellent | Good | Fair | Poor |
|--------|-----------|------|------|------|
| PSNR | > 35 dB | 30-35 dB | 25-30 dB | < 25 dB |
| SSIM | > 0.95 | 0.90-0.95 | 0.85-0.90 | < 0.85 |
| ΔE2000 | < 1.0 | 1.0-2.0 | 2.0-5.0 | > 5.0 |
| FVD | < 50 | 50-150 | 150-300 | > 300 |
| CLIP Aesthetic | > 0.7 | 0.6-0.7 | 0.4-0.6 | < 0.4 |
| DOVER (Overall) | > 0.6 | 0.4-0.6 | 0.2-0.4 | < 0.2 |
| FAST-VQA | > 0.65 | 0.5-0.65 | 0.3-0.5 | < 0.3 |

## Technical Documentation

See [doc.md](doc.md) for the complete theoretical framework.

## License

MIT License
