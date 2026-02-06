# ComfyUI Video Quality Metrics: Comprehensive Technical Reference

This document provides an exhaustive breakdown of every metric implemented in this library, covering their mathematical foundations, algorithmic logic, interpretation guidelines, and specific implications for Generative AI (AIGC).

---

## 📂 Evaluation Strategies: With vs. Without Reference

A common challenge in Generative AI (Diffusion models) is the lack of a "ground truth" reference video. This library supports three distinct evaluation modes:

### Scenario A: Video-to-Video / Upscaling (Full-Reference)
*   **Workflow**: You have an input video and an AI-processed output.
*   **Metrics**: Use **all metrics** (PSNR, SSIM, ΔE00, Warping Error, FVD).
*   **Goal**: Ensure the AI preserved the structure and color of the source while adding detail.

### Scenario B: Text-to-Video (No-Reference / Internal)
*   **Workflow**: You have a single generated video from a text prompt.
*   **Metrics**: Use **Temporal Consistency** metrics (`VQ_TemporalConsistency`, `VQ_MotionSmoothness`).
*   **Goal**: These nodes analyze the video *internally*. They detect flickering, jitter, and unnatural motion without needing a reference.

### Scenario C: Comparative Benchmarking (Workflow A vs. B)
*   **Workflow**: You have two different models/prompts and want to know which is better.
*   **Metrics**: Use **Distributional Metrics** (FVD, FID) and **Fidelity Metrics** (using one workflow as the "baseline").
*   **Goal**: Use `VQ_MetricsComparison` to see which workflow is statistically more realistic or stable.

---

## 🏗️ System Architecture
The following diagram illustrates the metric extraction and evaluation pipeline implemented in this toolkit:

```mermaid
graph TD
    subgraph Input_Layer ["Input Layer"]
        Gen_Vid["Generated Video [T, H, W, 3]"]
        Ref_Vid["Reference Video [T, H, W, 3]"]
    end

    subgraph Core_Metrics ["Core Metric Domains"]
        direction TB
        Fid["<b>Perceptual Fidelity</b><br/>PSNR, SSIM"]
        Temp["<b>Temporal Consistency</b><br/>Warping Error, Flickering"]
        Dist["<b>Distributional Realism</b><br/>FVD, FID (Pretrained)"]
        Col["<b>Color Accuracy</b><br/>CIEDE2000"]
    end

    subgraph Analysis_Layer ["Analysis & Reporting"]
        Stats["Statistical Significance<br/>(T-Test, Wilcoxon)"]
        Viz["Visualization<br/>(Radar Charts)"]
    end

    Input_Layer --> Core_Metrics
    Core_Metrics --> Analysis_Layer
```

The library is structured into modular components:
- **`core/`**: Pure PyTorch implementations of the math and feature extraction.
- **`nodes/`**: ComfyUI-specific wrapper classes.
- **`models/`**: Pre-trained weights management and auto-downloaders.
- **`utils/`**: Shared plotting, statistics, and tensor manipulation logic.

---

## 1. Perceptual & Signal Fidelity (Full-Reference)
These metrics require a "ground truth" reference video to compare against. They are ideal for evaluating **Upscalers**, **Video-to-Video** models, and **Motion Transfer** workflows.

### 1.1 Peak Signal-to-Noise Ratio (PSNR)
PSNR measures the ratio between the maximum possible power of a signal and the power of corrupting noise.
- **Calculation**: Derived from Mean Squared Error (MSE).
  $$MSE = \frac{1}{mn}\sum_{i=0}^{m-1}\sum_{j=0}^{n-1}[I(i,j) - K(i,j)]^2$$
  $$PSNR = 20 \cdot \log_{10}(MAX_I / \sqrt{MSE})$$
- **In this Library**: Implemented in `core/fidelity.py`. It handles batches by averaging MSE across all frames/images.
- **Interpretation**:
  - **> 40 dB**: Excellent (almost identical to human eye).
  - **30 - 40 dB**: Good (minor perceptible artifacts).
  - **< 20 dB**: Poor (significant degradation).
- **AIGC Context**: PSNR is very sensitive to pixel shifts. If your AI video is shifted by just 1 pixel, PSNR will tank even if the video looks perfect. Use it to detect noise and compression, not artistic quality.

### 1.2 Structural Similarity Index (SSIM)
SSIM is designed to align with the Human Visual System (HVS), which is sensitive to structure rather than pixel values.
- **Calculation**: Compares luminance ($l$), contrast ($c$), and structure ($s$).
  $$SSIM(x,y) = [l(x,y)]^\alpha \cdot [c(x,y)]^\beta \cdot [s(x,y)]^\gamma$$
- **In this Library**: Uses a Gaussian-weighted sliding window to compute local SSIM maps, which are then averaged. Non-standard input shapes are automatically handled via `tensor_to_hchw`.
- **Interpretation**: Range is [0, 1].
  - **> 0.95**: High structural preservation.
  - **< 0.80**: Noticeable structural warping or blurring.
- **AIGC Context**: Excellent for checking if an Image-to-Video model is "hallucinating" new backgrounds or changing the character's face structure too much.

---

## 2. Temporal Consistency (Motion Domain)
Temporal coherence is the hallmark of professional AI video. These metrics detect the "flicker" and "jitter" common in diffusion-based models.

### 2.1 Warping Error (Optical Flow Consistency)
Measures how consistently pixels move between frames.
- **Algorithm**:
  1. Calculate **Optical Flow** ($F_{t \to t+1}$) using **RAFT** (high accuracy) or **Lucas-Kanade** (fast fallback).
  2. Use the flow to **warp** frame $t+1$ back to the position of frame $t$.
  3. Calculate the difference (Residual) between the original frame $t$ and the warped version.
- **Occlusion Handling**: We compute bidirectional flow. If a pixel's movement doesn't match when calculated forward vs backward (Flow Consistency Check), we mask it out as an "occlusion" to avoid false positives.
- **Interpretation**: Lower is better. A sudden spike in warping error indicates a "jump" or "glitch" in the motion.

### 2.2 Temporal Flickering
Detects unstable lighting or intensity fluctuations.
- **Logic**: Analyzes the rolling standard deviation of the global mean brightness over a temporal window (default $T=5$).
- **Interpretation**: Range [0, 1].
  - **Low (< 0.1)**: Stable lighting.
  - **High (> 0.3)**: Distracting flickering (typical of early Video-SVD models without temporal attention).

### 2.3 Motion Smoothness (Jerk Analysis)
Measures the physical plausibility of motion trajectories.
- **Calculation**: Calculates the **Jerk** (the 2nd derivative of velocity/flow).
  $$Jerk = \frac{d^3s}{dt^3}$$
- **Interpretation**: 
  - **High Smoothness (> 0.8)**: Cinematic, steady motion.
  - **Low Smoothness (< 0.4)**: Robotic or jittery movement (high-frequency motion noise).

---

## 3. Distributional Realism (Reference-Free / Semantic)
These metrics answer the question: "Does this video *look* like a real video from the real world?"

### 3.1 Fréchet Video Distance (FVD)
The industry standard for evaluating video realism.
- **Mechanism**:
  1. Encodes videos into 512-dimensional features using an **R3D-18 (3D ResNet)** pretrained on the Kinetics-400 dataset.
  2. Measures the **Fréchet Distance** (Wasserstein-2) between the feature distributions of the generated set and a reference set.
- **Interpretation**: Lower is better. 0 means the distributions are identical.
- **Note**: Requires at least 2 videos in a batch to calculate covariance. For single videos, it provides a limited estimate.

### 3.2 Fréchet Inception Distance (FID)
Evaluates the aesthetic quality of individual frames.
- **Mechanism**: Uses **Inception V3** pretrained on ImageNet to extract 2048-dimensional features.
- **Use Case**: Detecting whether frames are "blurry" or "fake-looking" even if the motion is smooth.

### 3.3 Video Frame-by-Frame FID (VideoFID)
A hybrid metric for video-to-video comparison.
- **Logic**: Calculates the FID between the generated video's frames and the reference video's frames.
- **Why it matters**: Unlike global FID (which pools all frames), this measures how well the AI reproduces the *specific* aesthetics of each frame in order.
- **Output**: Mean FID and per-frame distance range.

---

## 4. Deep Learning No-Reference Metrics (SOTA)
These advanced metrics use neural networks trained on human subjective ratings (Mean Opinion Scores) to judge quality without needing a reference video. They are the **gold standard** for evaluating Text-to-Video models.

### 4.1 CLIP-IQA (Aesthetic & Alignment)
Leverages OpenAI's CLIP model (ViT-B/32) to measure visual appeal and prompt adherence.
- **Aesthetic Score**: measures the "artistic beauty" of frames.
- **Text-Video Alignment**: Measures how well the video matches the text prompt.
- **Temporal Drift**: Detects if the video typically "forgets" the prompt over time (e.g., a cat turning into a dog).

### 4.2 DOVER (Disentangled Objective Video Quality Evaluator)
A state-of-the-art method that separates quality into two distinct axes:
1.  **Aesthetic Score**: Composition, color harmony, lighting, and "artistic vibe".
2.  **Technical Score**: Sharpness, noise, compression artifacts, and exposure issues.
-   **Architecture**: Uses a **Swin Transformer (Tiny)** backbone.
-   **Why use it**: Tells you *why* a video is bad (e.g., "Good composition but too blurry").

### 4.3 FAST-VQA (Fragment Attention Network)
Designed for efficiency and high-resolution video assessment.
-   **Mechanism**: Uses **Grid Mini-patch Sampling (GMS)**. Instead of resizing a 4K video to 224x224 (destroying detail), it samples small 32x32 patches at full resolution.
-   **Architecture**: Fragment Attention Network (FANet).
-   **Use Case**: Best for detecting fine-grained artifacts like noise or upscaling glitches.

---

## 4. Reporting & Workflow Analysis

### 4.1 Metrics Logger (JSON)
Designed for automation and experiment tracking.
- **Output**: A structured JSON string capturing every metric value.
- **Use Case**: Save these outputs to a text file using ComfyUI's standard Save nodes to build a long-term benchmark database of your various workflows.

### 4.2 Workflow Comparison & Statistical Power
Compares two workflows (A and B) and highlights the "Winner" for each metric.
- **Statistical Significance**: Uses the P-value to tell you if a +2dB improvement in PSNR is actually meaningful or just random chance.
- **Rule of Thumb**: Only trust an improvement if the P-value is **below 0.05**.

---

## 5. Color & Perceptual Stability

### 5.1 CIEDE2000 ($\Delta E_{00}$)
The most advanced metric for delta-color difference.
- **Logic**: Converts RGB to **CIELAB** space, which is designed to be perceptually linear. It applies specialized weights to compensate for the HVS sensitivity to different hues (like blue vs red).
- **Interpretation**:
  - **< 1.0**: Imperceptible change.
  - **2.3**: "Noticeable difference" threshold.
  - **> 5.0**: Significant color drift (e.g., skin tones shifting to green).

---

## 6. Statistical Tools
How to tell if **Workflow A** is definitively better than **Workflow B**.

### 6.1 T-Test (Independent/Dependent)
- **Use Case**: Comparing average scores across two different experiments.
- **P-Value**: If $P < 0.05$, there is a 95% probability that the difference between your workflows is real and not just a lucky seed.

### 6.2 Wilcoxon Signed-Rank Test
- **Use Case**: Used for paired comparisons (e.g., comparing Workflow A and B on the same 50 prompts). It is more robust than the T-test if your data has "outliers" (one extremely bad video that ruins the average).

---

## 📊 Evaluation Strategy: The Radar Chart
Because no single metric is perfect, we recommend using the **Radar Chart Node**.
- **Spade Shape**: Good for upscaling (High PSNR, High SSIM, Low warping).
- **Circle Shape**: Good for creative generation (Low FVD, High Smoothness, ignoring PSNR).

For a complete analysis, connect all metric outputs to the `VQ_RadarChart` node to visualize your model's "Performance Fingerprint."
