import numpy as np
import scipy.ndimage as ndimage
import tifffile
import pandas as pd
from pathlib import Path
from tifffile import TiffFile
from ome_types import from_xml
from skimage.feature import peak_local_max
from skimage.morphology import disk, dilation
from skimage.segmentation import watershed
from skimage.measure import regionprops
from scipy.spatial.distance import cdist, squareform
from scipy.cluster.hierarchy import linkage, optimal_leaf_ordering, leaves_list, fcluster
from scipy.signal import find_peaks, peak_prominences, peak_widths, coherence, welch, fftconvolve


def get_sampling_rate_hz(ome_tif_path):
    with tifffile.TiffFile(ome_tif_path) as tf:
        ome_xml = tf.ome_metadata
        if not ome_xml:
            ome_xml = tf.pages[0].description

    ome = from_xml(ome_xml)
    pixels = ome.images[0].pixels

    ti = pixels.time_increment
    if ti is not None:
        dt_s = float(getattr(ti, "value", ti))
        return 1.0 / dt_s

    dts = []
    for pl in (pixels.planes or []):
        if pl.delta_t is not None:
            dts.append(float(getattr(pl.delta_t, "value", pl.delta_t)))

    if len(dts) >= 2:
        dts = np.array(dts)
        dt = np.median(np.diff(np.sort(dts)))
        return 1.0 / dt

    return None


def make_temporal_kernel(fs, tau_rise=0.15, tau_decay=0.8, length_s=2.5):
    L = int(np.ceil(length_s * fs))
    t = np.arange(L) / fs
    k = np.exp(-t / tau_decay) - np.exp(-t / tau_rise)
    k = k - k.mean()
    k = k / (np.linalg.norm(k) + 1e-12)
    return k.astype(np.float32)


def spatial_bandpass_chunk(chunk, sigma_small=1.2, sigma_large=6.0):
    g1 = ndimage.gaussian_filter(chunk, sigma=(0, sigma_small, sigma_small), mode="reflect")
    g2 = ndimage.gaussian_filter(chunk, sigma=(0, sigma_large, sigma_large), mode="reflect")
    return (g1 - g2).astype(np.float32)


def robust_frame_normalize(chunk):
    med = np.median(chunk, axis=(1, 2), keepdims=True)
    mad = np.median(np.abs(chunk - med), axis=(1, 2), keepdims=True) + 1e-6
    return ((chunk - med) / mad).astype(np.float32)


def combine_running_stats(mean, M2, n, mean_c, var_c, m):
    if n == 0:
        return mean_c, var_c * (m - 1), m
    delta = mean_c - mean
    n_new = n + m
    mean_new = mean + delta * (m / n_new)
    M2_c = var_c * (m - 1)
    M2_new = M2 + M2_c + (delta * delta) * (n * m / n_new)
    return mean_new, M2_new, n_new


def activity_image_stream(video, fs, hp_s=2.0, chunk_T=256, rectify=True):
    T, H, W = video.shape
    win = int(max(3, round(hp_s * fs)))
    S = np.zeros((H, W), dtype=np.float32)
    t0 = 0
    while t0 < T:
        t1 = min(T, t0 + chunk_T)
        chunk = video[t0:t1].astype(np.float32, copy=False)
        baseline = ndimage.uniform_filter1d(chunk, size=win, axis=0, mode="nearest")
        hp = chunk - baseline
        if rectify:
            hp = np.maximum(hp, 0)
        S += (hp * hp).sum(axis=0)
        t0 = t1
    S = np.sqrt(S / T)
    return S


def dog2d(img, sigma_small=1.2, sigma_large=6.0):
    g1 = ndimage.gaussian_filter(img, sigma=sigma_small, mode="reflect")
    g2 = ndimage.gaussian_filter(img, sigma=sigma_large, mode="reflect")
    return (g1 - g2).astype(np.float32)


def segment_rois_from_score2d(score, min_distance=6, seed_thresh=6.0, mask_thresh=3.5,
                              min_area=40, max_area=2500, max_eccentricity=0.97):
    coords = peak_local_max(score, min_distance=min_distance, threshold_abs=seed_thresh)
    markers = np.zeros(score.shape, dtype=np.int32)
    for i, (r, c) in enumerate(coords, start=1):
        markers[r, c] = i
    mask = score > mask_thresh
    labels_ws = watershed(-score, markers=markers, mask=mask)
    labels_clean = np.zeros_like(labels_ws, dtype=np.int32)
    out_id = 1
    for reg in regionprops(labels_ws):
        if reg.area < min_area or reg.area > max_area:
            continue
        if reg.eccentricity > max_eccentricity:
            continue
        labels_clean[labels_ws == reg.label] = out_id
        out_id += 1
    return labels_clean


def make_neuropil_masks(roi_labels, inner_r=3, outer_r=8):
    all_rois = roi_labels > 0
    neuropil = np.zeros_like(roi_labels, dtype=np.int32)
    se_in, se_out = disk(inner_r), disk(outer_r)
    roi_ids = np.unique(roi_labels)
    roi_ids = roi_ids[roi_ids > 0]
    for rid in roi_ids:
        roi = roi_labels == rid
        inner = dilation(roi, se_in)
        outer = dilation(roi, se_out)
        ring = outer & ~inner
        ring = ring & ~all_rois
        neuropil[ring] = rid
    return neuropil


def extract_traces(video, roi_labels, neuropil_labels, neuropil_scale=0.7):
    T = video.shape[0]
    roi_ids = np.unique(roi_labels)
    roi_ids = roi_ids[roi_ids > 0]
    F_roi = np.zeros((len(roi_ids), T), dtype=np.float32)
    F_np = np.zeros((len(roi_ids), T), dtype=np.float32)
    v = video.astype(np.float32, copy=False)
    for i, rid in enumerate(roi_ids):
        roi = roi_labels == rid
        np_mask = neuropil_labels == rid
        F_roi[i] = v[:, roi].mean(axis=1)
        F_np[i] = v[:, np_mask].mean(axis=1) if np_mask.any() else 0.0
    F_corr = F_roi - neuropil_scale * F_np
    return roi_ids, F_corr, F_roi, F_np


def dff_from_traces(F_roi, F_np=None, fs=30.0, neuropil_scale=0.7,
                    baseline_win_s=60.0, baseline_percentile=10.0, f0_floor=1.0):
    F_roi = F_roi.astype(np.float32, copy=False)
    if F_np is not None:
        F_np = F_np.astype(np.float32, copy=False)
        Fcorr = F_roi - neuropil_scale * F_np
    else:
        Fcorr = F_roi
    win = int(round(baseline_win_s * fs))
    win = max(3, win)
    if win % 2 == 0:
        win += 1
    F0 = ndimage.percentile_filter(Fcorr, percentile=baseline_percentile,
                                   size=(1, win), mode="nearest").astype(np.float32)
    F0 = np.maximum(F0, f0_floor)
    dff = (Fcorr - F0) / F0
    return dff, F0, Fcorr


def matched_filter_traces(F, k_t):
    F0 = F - F.mean(axis=1, keepdims=True)
    resp = np.stack([fftconvolve(f, k_t[::-1], mode="same") for f in F0]).astype(np.float32)
    return np.maximum(resp, 0)


def compute_xcorr_lag_matrix(dff, max_lag=500):
    X = np.asarray(dff)
    X = np.nan_to_num(X)
    X = (X - np.mean(X, axis=1, keepdims=True)) / (np.std(X, axis=1, keepdims=True) + 1e-12)
    n, T = X.shape
    X0 = X - X.mean(axis=1, keepdims=True)
    std = X0.std(axis=1, keepdims=True) + 1e-12
    Z = X0 / std
    nfft = 1 << int(np.ceil(np.log2(2 * T - 1)))
    F = np.fft.rfft(Z, n=nfft, axis=1)

    full_lags = np.arange(-(T - 1), T)
    if max_lag is not None:
        max_lag = int(max_lag)
        keep = (full_lags >= -max_lag) & (full_lags <= max_lag)
        lag_mask = keep
        lags = full_lags[keep]
    else:
        lag_mask = slice(None)
        lags = full_lags

    L = np.zeros((n, n), dtype=int)
    M = np.zeros((n, n), dtype=float)

    for i in range(n):
        cps = F[i][None, :] * np.conj(F)
        r = np.fft.irfft(cps, n=nfft, axis=1)
        r = np.concatenate([r[:, -(T - 1):], r[:, :T]], axis=1)
        r = r / T
        r_sel = r[:, lag_mask]
        k = np.argmax(r_sel, axis=1)
        L[i, :] = lags[k]
        M[i, :] = r_sel[np.arange(n), k]

    C = np.corrcoef(X)
    return L.astype(np.float32), M, C


def functional_affinity(L, M, tau_lag=50, use_absM=True, drop_negative=True, eps=1e-12):
    L = np.asarray(L)
    M = np.asarray(M)
    S = np.abs(M) if use_absM else M.copy()
    if drop_negative:
        S = np.clip(S, 0.0, 1.0)
    else:
        S = np.clip(S, -1.0, 1.0)
    P = np.exp(-(np.abs(L) / (tau_lag + eps)) ** 2)
    Wt = S * P
    Wt = 0.5 * (Wt + Wt.T)
    np.fill_diagonal(Wt, 0.0)
    return Wt


def spatial_affinity(D, sigma_pix=None, eps=1e-12):
    D = np.asarray(D)
    if sigma_pix is None:
        nn = np.partition(D + np.eye(D.shape[0]) * 1e9, 1, axis=1)[:, 1]
        sigma_pix = np.median(nn)
    Ws = np.exp(-(D / (sigma_pix + eps)) ** 2)
    np.fill_diagonal(Ws, 0.0)
    return Ws, sigma_pix


def combined_affinity(Wt, Ws, lam=0.7, beta=0.5):
    W = lam * Wt + (1 - lam) * Ws
    W = 0.5 * (W + W.T)
    return W


def hclust_ordering(W):
    Wn = W / (W.max() + 1e-12)
    dist = 1.0 - Wn
    Z = linkage(squareform(dist, checks=False), method="average")
    Zopt = optimal_leaf_ordering(Z, squareform(dist, checks=False))
    order = leaves_list(Zopt)
    labels = np.empty_like(order)
    labels[order] = np.arange(1, W.shape[0] + 1)
    return order, labels


def hclust_auto_clusters_from_affinity(W, method="average"):
    Wn = W / (W.max() + 1e-12)
    dist = 1.0 - Wn
    Z = linkage(squareform(dist, checks=False), method=method)
    d = Z[:, 2]
    jumps = np.diff(d)
    j = np.argmax(jumps)
    t = 0.5 * (d[j] + d[j + 1])
    labels = fcluster(Z, t=t, criterion="distance")
    return labels, Z, t


def extract_peak_features(dff, prominence_thresh=0.1):
    parameters = [
        'n_peaks',
        'mean_isi_frames', 'std_isi_frames', 'max_isi_frames', 'min_isi_frames',
        'mean_prominence', 'std_prominence', 'max_prominence', 'min_prominence',
        'min_width_frames', 'max_width_frames', 'mean_width_frames', 'std_width_frames'
    ]
    parameters_np = np.zeros((dff.shape[0], len(parameters)), dtype=np.float32)
    peak_indexes = []

    for i, trace in enumerate(dff):
        peaks, _ = find_peaks(trace, prominence=prominence_thresh)
        n_peaks = len(peaks)
        if n_peaks == 0:
            parameters_np[i] = np.nan
            parameters_np[i, 0] = 0
            peak_indexes.append(peaks)
            continue

        prominences, left_bases, right_bases = peak_prominences(trace, peaks)
        widths = peak_widths(trace, peaks, prominence_data=(prominences, left_bases, right_bases))[0]
        has_multi = n_peaks > 1

        parameters_np[i] = np.array([
            n_peaks,
            np.mean(np.diff(peaks)) if has_multi else np.nan,
            np.std(np.diff(peaks)) if has_multi else np.nan,
            np.max(np.diff(peaks)) if has_multi else np.nan,
            np.min(np.diff(peaks)) if has_multi else np.nan,
            np.mean(prominences), np.std(prominences),
            np.max(prominences), np.min(prominences),
            np.min(widths), np.max(widths),
            np.mean(widths), np.std(widths)
        ])
        peak_indexes.append(peaks)

    return parameters_np, parameters, peak_indexes


def csd_matrix_welch(X, fs=1.0, nperseg=512, noverlap=256, window="hann", detrend=True):
    X = np.asarray(X)
    N, T = X.shape
    step = nperseg - noverlap
    nwin = 1 + (T - nperseg) // step

    w = np.hanning(nperseg)
    U = (w ** 2).sum()
    scale = 1.0 / (fs * U)
    F = nperseg // 2 + 1
    S = np.zeros((F, N, N), dtype=np.complex128)

    for k in range(nwin):
        sl = slice(k * step, k * step + nperseg)
        seg = X[:, sl].copy()
        if detrend:
            seg -= seg.mean(axis=1, keepdims=True)
        seg *= w[None, :]
        Xf = np.fft.rfft(seg, axis=1)
        S += np.einsum("nf,mf->fmn", Xf, np.conj(Xf))

    S *= scale / nwin

    if nperseg % 2 == 0:
        S[1:-1] *= 2.0
    else:
        S[1:] *= 2.0

    f = np.fft.rfftfreq(nperseg, d=1 / fs)
    P = np.real(np.diagonal(S, axis1=1, axis2=2))
    denom = P[:, :, None] * P[:, None, :]
    Cmat = (np.abs(S) ** 2) / (denom + 1e-30)

    GC = np.empty(F)
    for fi in range(F):
        evals = np.linalg.eigvalsh(S[fi])
        tr = np.real(evals.sum())
        GC[fi] = (np.real(evals[-1]) / tr) if tr > 0 else np.nan

    return f, S, Cmat, GC


def compute_coherence_features(dff, fs=1.0, nperseg=512, noverlap=256):
    ref = dff.mean(axis=0)
    f, P_ref = welch(ref, fs=fs, nperseg=nperseg, noverlap=noverlap, window="hann")

    C = []
    for x in dff:
        _, cxy = coherence(x, ref, fs=fs, nperseg=nperseg, noverlap=noverlap, window="hann")
        C.append(cxy)
    C = np.vstack(C)

    band = (f > 0) & (f <= 0.5)
    w = P_ref[band].copy()
    w = w / (w.sum() + 1e-12)
    score_per_trace = (C[:, band] * w).sum(axis=1)

    f_csd, S, Cmat, GC = csd_matrix_welch(dff, fs=fs, nperseg=nperseg, noverlap=noverlap)
    band_csd = (f_csd > 0) & (f_csd <= 0.5)

    w_csd = np.real(np.trace(S[band_csd], axis1=1, axis2=2))
    w_csd = w_csd / (w_csd.sum() + 1e-30)

    C_band = np.tensordot(w_csd, Cmat[band_csd], axes=(0, 0))
    S_band = np.tensordot(w_csd, S[band_csd], axes=(0, 0))
    GC_band = np.nansum(w_csd * GC[band_csd])

    return score_per_trace, C_band, S_band, GC_band


def save_results(tifpath, video, total_varframe, score2d, roi_labels, neuropil_labels,
                 F_roi, F_np, F_corr, dff, labels, peak_indexes, parameters_np, parameters,
                 C, L, M, D, W, score_per_trace, C_band, S_band):
    base_path = Path(tifpath)
    out_dir = base_path.parent / f"{base_path.stem}_results"
    out_dir.mkdir(parents=True, exist_ok=True)

    def save_tiff(name, arr, dtype=None):
        a = np.asarray(arr)
        if dtype is not None:
            a = a.astype(dtype, copy=False)
        tifffile.imwrite(str(out_dir / name), a, photometric="minisblack",
                         compression="zlib", metadata=None)

    save_tiff("first_frame.tif", video[0], dtype=np.float32)
    save_tiff("varframe.tif", total_varframe, dtype=np.float32)
    save_tiff("score2d.tif", score2d, dtype=np.float32)
    save_tiff("roi_labels.tif", roi_labels, dtype=np.int32)
    save_tiff("neuropil_labels.tif", neuropil_labels, dtype=np.int32)

    xlsx_path = out_dir / "traces.xlsx"
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        pd.DataFrame(F_roi.T).to_excel(writer, sheet_name="ROI_Raw_Traces", index=False)
        pd.DataFrame(F_np.T).to_excel(writer, sheet_name="Neuropil_Raw_Traces", index=False)
        pd.DataFrame(F_corr.T).to_excel(writer, sheet_name="Corrected_Traces", index=False)
        pd.DataFrame(dff.T).to_excel(writer, sheet_name="dF_F_Traces", index=False)
        pd.DataFrame(labels, columns=["Cluster_Label"]).to_excel(writer, sheet_name="Clustering_Labels", index=False)
        pd.DataFrame(peak_indexes).to_excel(writer, sheet_name="Peak_Indexes", index=False)
        pd.DataFrame(parameters_np, columns=parameters).to_excel(writer, sheet_name="Peak_Parameters", index=False)

    xlsx_path = out_dir / "correlations.xlsx"
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        pd.DataFrame(C).to_excel(writer, sheet_name="Correlation_Matrix", index=False)
        pd.DataFrame(L).to_excel(writer, sheet_name="Lag_Matrix", index=False)
        pd.DataFrame(M).to_excel(writer, sheet_name="Max_Corr_at_lag_Matrix", index=False)
        pd.DataFrame(D).to_excel(writer, sheet_name="Distance_Matrix", index=False)
        pd.DataFrame(W).to_excel(writer, sheet_name="Affinity_Matrix", index=False)

    xlsx_path = out_dir / "coherence.xlsx"
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        pd.DataFrame(score_per_trace).to_excel(writer, sheet_name="Band_Averaged_Coherence", index=False)
        pd.DataFrame(C_band).to_excel(writer, sheet_name="CSD_Averaged_Coherence_Matrix", index=False)
        pd.DataFrame(np.real(S_band)).to_excel(writer, sheet_name="CSD_Real_Matrix", index=False)
        pd.DataFrame(np.imag(S_band)).to_excel(writer, sheet_name="CSD_Imag_Matrix", index=False)
        pd.DataFrame(np.abs(S_band)).to_excel(writer, sheet_name="CSD_Magnitude_Matrix", index=False)

    return out_dir
