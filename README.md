# Organoid Calcium Imaging Pipeline

Automated ROI detection, trace extraction, and functional connectivity analysis for calcium imaging recordings.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

See `example.ipynb` for a full walkthrough. The core functions are in `pipeline.py`.

```python
from pipeline import (
    get_sampling_rate_hz, activity_image_stream, dog2d,
    segment_rois_from_score2d, make_neuropil_masks, extract_traces,
    dff_from_traces, compute_xcorr_lag_matrix, functional_affinity,
    spatial_affinity, combined_affinity, hclust_auto_clusters_from_affinity,
    extract_peak_features, compute_coherence_features, save_results
)
```

## Pipeline Overview

1. **Activity mapping** — High-pass filtered RMS activity image, refined with a difference-of-Gaussians spatial bandpass and robust z-scoring.
2. **ROI segmentation** — Watershed segmentation seeded from local maxima of the score map, filtered by area and eccentricity.
3. **Neuropil subtraction** — Annular neuropil masks (excluding all ROI pixels) per ROI; corrected fluorescence = F_roi − 0.7 × F_neuropil.
4. **ΔF/F₀** — Running 10th-percentile baseline (60 s window) on neuropil-corrected traces.
5. **Peak features** — Peak detection with prominence thresholding; ISI, prominence, and width statistics per ROI.
6. **Cross-correlation & lag** — FFT-based pairwise normalized cross-correlation; lag and magnitude at maximum correlation.
7. **Affinity & clustering** — Combined functional (lag-penalized cross-correlation) and spatial (Gaussian distance kernel) affinity matrix; agglomerative clustering with automatic threshold selection.
8. **Coherence** — Per-ROI magnitude-squared coherence with the population mean; Welch-based cross-spectral density matrix; band-averaged coherence and global coherence.

## Materials and Methods

Regions of interest (ROIs) were automatically detected from calcium imaging recordings stored as OME-TIFF files. First, a two-dimensional activity score map was computed by calculating the root-mean-square amplitude of the high-pass filtered video (uniform-filter baseline, 2 s window) at each pixel, followed by a difference-of-Gaussians spatial bandpass (σ_small = 1.0 px, σ_large = 4.0 px) and robust z-scoring using the median and median absolute deviation. ROIs were segmented from this score map using seeded watershed segmentation: local maxima above a z-score threshold of 6.0 (minimum distance 6 px) served as seeds, and the watershed was restricted to pixels exceeding a mask threshold of 3.5. Candidate ROIs were filtered by area (40–2500 px²) and eccentricity (<0.97).

For each ROI, a neuropil annulus was defined by morphologically dilating the ROI mask with disk-shaped structuring elements (inner radius 3 px, outer radius 8 px) and excluding all ROI-occupied pixels. Mean fluorescence traces were extracted from each ROI and its corresponding neuropil region. Neuropil-corrected fluorescence was computed as F_corr = F_ROI − 0.7 × F_neuropil, and ΔF/F₀ was calculated using a running 10th-percentile baseline over a 60-second sliding window, with a minimum baseline floor of 1.0 to avoid division instabilities. Calcium transient peaks were detected on the ΔF/F₀ traces using a prominence threshold of 0.1, and summary statistics (inter-spike intervals, prominences, and half-widths) were extracted for each ROI.

Pairwise functional connectivity was assessed using FFT-based normalized cross-correlation, yielding for each ROI pair both the zero-lag Pearson correlation and the lag (in frames) at which the normalized cross-correlation was maximal (within ±500 frames). A combined affinity matrix was constructed by blending a functional affinity component — the absolute peak cross-correlation attenuated by a Gaussian lag penalty (τ = 50 frames) — with a spatial affinity component based on a Gaussian kernel over pairwise Euclidean distances between ROI centroids (weight λ = 0.7 for functional, 0.3 for spatial). ROIs were grouped into functional clusters by agglomerative hierarchical clustering (complete linkage) on the affinity-derived distance matrix, with the number of clusters determined automatically by identifying the largest gap in the dendrogram merge distances. Spectral coherence was quantified per ROI as the power-weighted mean magnitude-squared coherence with the population-average trace, computed via Welch's method (Hann window, segment length 512 frames, 50% overlap). A full cross-spectral density matrix was also estimated and band-averaged to yield pairwise coherence and global coherence metrics.

## Outputs

For each recording, the pipeline saves to a `<filename>_results/` subfolder:

- `first_frame.tif`, `varframe.tif`, `score2d.tif` — reference images
- `roi_labels.tif`, `neuropil_labels.tif` — segmentation masks
- `traces.xlsx` — raw, corrected, and ΔF/F₀ traces; cluster labels; peak indices and features
- `correlations.xlsx` — correlation, lag, max-correlation, distance, and affinity matrices
- `coherence.xlsx` — per-ROI coherence scores and cross-spectral density matrices

# NOTES:

- Correlation tells us that the regions in our videos behave similarly as they “co-fluctuate”, while the coherence metrics tell us that the regions share a genuine oscillatory process at specific frequencies, which is a much stronger claim about the organoid functional connectivity.
- In the sheet are have tabs called “Band Averaged Coherence”, “CSD Averaged Coherence Matrix”, “CSD Real Matrix”, “CSD Imag Matrix”, and “CSD Magnitude Matrix”.*
    - The “Band Averaged Coherence” tells us which individual ROIs are tightly locked to the ensemble rhythm vs. which are doing their own thing. High-coherence ROIs are our "network participants"; low-coherence ones may be isolated or noisy.
    - The “CSD Averaged Coherence Matrix” is like a correlation matrix but frequency-resolved, so it's insensitive to slow drift and baseline wander that inflate Pearson correlations. Two ROIs can have low Pearson r but high coherence if they share oscillatory structure at a consistent frequency with a phase offset.
    - The “CSD Real Matrix”, “CSD Imag Matrix”, and “CSD Magnitude Matrix” tabs are the real, imaginary, and magnitude of the complex cross spectral density matrix, and they give us the magnitude of the real and imaginary components as well as the modulus of the complex number. They can be interpreted as giving us both magnitude (real) and phase (imag) relationships between every ROI pair. The real part capturing the in-phase coupling, while the imaginary part captures lagged/leading coupling.


\* NOTE: these matrices are frequency collapsed in the saved xlsx.

