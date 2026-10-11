"""pupilkit.py -- minimal, dependency-light tools for pupil data.

Requires only NumPy and SciPy. Written for teaching: every step is
explicit so that you can inspect, change and test it.
"""
import numpy as np
from scipy import signal, stats

# ----------------------------------------------------------------------
# 1. Blink detection and interpolation
# ----------------------------------------------------------------------
def detect_blinks(pupil, fs, vel_thresh=None, pad_ms=100):
    """Return a boolean mask of samples to treat as missing.

    pupil      : 1-D array of pupil size (0 or NaN where tracking was lost)
    fs         : sampling rate in Hz
    vel_thresh : speed (units per second) above which a sample is
                 implausible; default = median + 5 * MAD of the speed
    pad_ms     : margin added before and after each blink
    """
    p = np.asarray(pupil, dtype=float).copy()
    bad = ~np.isfinite(p) | (p <= 0)
    speed = np.abs(np.gradient(np.where(bad, np.nan, p))) * fs
    if vel_thresh is None:
        s = speed[np.isfinite(speed)]
        mad = np.median(np.abs(s - np.median(s)))
        vel_thresh = np.median(s) + 5 * 1.4826 * mad
    bad |= np.nan_to_num(speed, nan=0.0) > vel_thresh
    pad = int(round(pad_ms / 1000 * fs))
    if pad > 0 and bad.any():
        kernel = np.ones(2 * pad + 1)
        bad = np.convolve(bad.astype(float), kernel, mode="same") > 0
    return bad


def interpolate_missing(pupil, bad, max_gap_s=None, fs=None):
    """Linearly interpolate samples flagged in `bad`.

    Gaps longer than max_gap_s (if given) are left as NaN.
    """
    p = np.asarray(pupil, dtype=float).copy()
    p[bad] = np.nan
    idx = np.arange(len(p))
    good = np.isfinite(p)
    if good.sum() < 2:
        return p
    out = np.interp(idx, idx[good], p[good])
    if max_gap_s is not None:
        # restore NaN in long gaps
        edges = np.diff(np.r_[0, bad.astype(int), 0])
        starts, ends = np.where(edges == 1)[0], np.where(edges == -1)[0]
        for s, e in zip(starts, ends):
            if (e - s) / fs > max_gap_s:
                out[s:e] = np.nan
    return out


# ----------------------------------------------------------------------
# 2. Filtering and downsampling
# ----------------------------------------------------------------------
def lowpass(x, fs, cutoff=4.0, order=3):
    """Zero-phase Butterworth low-pass; NaN gaps are bridged, then restored."""
    x = np.asarray(x, dtype=float); gap = np.isnan(x); i = np.arange(len(x))
    b, a = signal.butter(order, cutoff / (fs / 2), btype="low")
    out = signal.filtfilt(b, a, np.interp(i, i[~gap], x[~gap]))
    out[gap] = np.nan
    return out


def downsample(x, fs, new_fs):
    """Average valid samples in bins (empty: NaN); fs must be a multiple of new_fs."""
    f = int(fs // new_fs)
    if f * new_fs != fs: raise ValueError(f"{fs} Hz is not a multiple of {new_fs} Hz")
    bins = x[: len(x) // f * f].reshape(-1, f)
    n = np.isfinite(bins).sum(axis=1)
    return np.where(n > 0, np.nansum(bins, axis=1) / np.maximum(n, 1), np.nan)


# ----------------------------------------------------------------------
# 3. Epoching and baseline correction
# ----------------------------------------------------------------------
def epoch(x, fs, onsets_s, tmin=-0.5, tmax=3.0):
    """Cut epochs around event onsets (in seconds). Returns (times, array)."""
    i0, i1 = int(round(tmin * fs)), int(round(tmax * fs))
    times = np.arange(i0, i1) / fs
    rows = []
    for t in onsets_s:
        c = int(round(t * fs))
        if c + i0 >= 0 and c + i1 <= len(x):
            rows.append(x[c + i0 : c + i1])
    return times, np.array(rows)


def baseline_correct(epochs, times, window=(-0.2, 0.0), method="subtractive"):
    """Subtractive (default) or divisive baseline correction."""
    m = (times >= window[0]) & (times < window[1])
    v = np.isfinite(epochs[:, m]).sum(axis=1, keepdims=True)   # no baseline: NaN
    base = np.nansum(epochs[:, m], axis=1, keepdims=True) / np.where(v, v, np.nan)
    if method == "subtractive":
        return epochs - base
    if method == "divisive":
        return (epochs - base) / base * 100.0   # percent change
    raise ValueError("method must be 'subtractive' or 'divisive'")


def reject_epochs(epochs, times, max_missing=0.25, window=(-0.2, 0.0), min_base=0.5):
    """Drop epochs with > max_missing NaN or < min_base valid baseline."""
    m = (times >= window[0]) & (times < window[1])
    keep = np.isnan(epochs).mean(1) <= max_missing
    return epochs[keep & (np.isfinite(epochs[:, m]).mean(1) >= min_base)]


# ----------------------------------------------------------------------
# 4. Light-reflex parameters
# ----------------------------------------------------------------------
def plr_parameters(trace, fs, onset_s=0.0, base_win=0.5):
    """Basic light-reflex parameters from one baseline-included trace.

    The trace must start base_win seconds before the flash at onset_s.
    Returns a dict with baseline, minimum, amplitude, latency,
    maximum constriction velocity and time to minimum.
    """
    t = np.arange(len(trace)) / fs - base_win + onset_s
    base = trace[t < onset_s].mean()
    post = t >= onset_s
    tmin_i = np.argmin(np.where(post, trace, np.inf))
    vel = np.gradient(trace) * fs
    mcv_i = np.argmin(np.where(post & (t <= t[tmin_i]), vel, np.inf))
    # latency: first post-onset sample where velocity < -0.5 mm/s (constriction)
    cand = np.where(post & (vel < -0.5))[0]
    lat = t[cand[0]] - onset_s if cand.size else np.nan
    return dict(baseline=base, minimum=trace[tmin_i],
                amplitude=base - trace[tmin_i], latency=lat,
                max_constriction_velocity=-vel[mcv_i],
                time_to_min=t[tmin_i] - onset_s)


# ----------------------------------------------------------------------
# 5. Pupillary unrest index (simplified, after Luedtke et al., 1998)
# ----------------------------------------------------------------------
def pupillary_unrest_index(pupil_mm, fs, seg_s=0.64):
    """Sum of absolute changes between means of consecutive short
    segments, expressed in mm per minute."""
    n = int(round(seg_s * fs))
    k = len(pupil_mm) // n
    means = pupil_mm[: k * n].reshape(k, n).mean(axis=1)
    minutes = k * n / fs / 60.0
    return np.sum(np.abs(np.diff(means))) / minutes


# ----------------------------------------------------------------------
# 6. Pupil response function and deconvolution
# ----------------------------------------------------------------------
def pupil_response_function(fs, n=10.1, tmax=0.93, dur=4.0):
    """Erlang gamma kernel of Hoeks and Levelt (1993), peak = 1."""
    t = np.arange(0, dur, 1 / fs)
    h = t ** n * np.exp(-n * t / tmax)
    return h / h.max()


def deconvolve(x, fs, onsets_by_condition, n=10.1, tmax=0.93):
    """Estimate one response amplitude per condition by least squares.

    x                   : continuous, baseline-free pupil signal
    onsets_by_condition : dict {name: list of onset times in s}
    Returns dict {name: amplitude} plus an 'intercept' term.
    """
    h = pupil_response_function(fs, n, tmax)
    names = list(onsets_by_condition)
    X = np.ones((len(x), len(names) + 1))
    for j, name in enumerate(names):
        stick = np.zeros(len(x))
        for t in onsets_by_condition[name]:
            i = int(round(t * fs))
            if i < len(x):
                stick[i] = 1.0
        X[:, j] = np.convolve(stick, h)[: len(x)]
    ok = np.isfinite(x)                      # skip samples left missing
    beta, *_ = np.linalg.lstsq(X[ok], x[ok], rcond=None)
    out = {nm: b for nm, b in zip(names, beta[:-1])}
    out["intercept"] = beta[-1]
    return out


# ----------------------------------------------------------------------
# 7. Cluster-based permutation test (paired, one-dimensional)
# ----------------------------------------------------------------------
def cluster_permutation_test(a, b, n_perm=2000, alpha=0.05, seed=0):
    """Paired cluster-based permutation test over time.

    a, b : arrays (participants x time) for two conditions.
    Returns list of (start_index, end_index, cluster_mass, p_value).
    """
    rng = np.random.default_rng(seed)
    d = a - b
    n = d.shape[0]
    t_crit = stats.t.ppf(1 - alpha / 2, n - 1)

    def clusters(tvals):
        found, mass = [], []
        above = np.abs(tvals) > t_crit
        sign = np.sign(tvals)
        i = 0
        while i < len(tvals):
            if above[i]:
                j = i
                while j + 1 < len(tvals) and above[j + 1] and sign[j + 1] == sign[i]:
                    j += 1
                found.append((i, j))
                mass.append(np.sum(tvals[i : j + 1]))
                i = j + 1
            else:
                i += 1
        return found, mass

    def tstat(x):
        return x.mean(0) / (x.std(0, ddof=1) / np.sqrt(len(x)))

    obs_clusters, obs_mass = clusters(tstat(d))
    null_max = np.zeros(n_perm)
    for k in range(n_perm):
        flips = rng.choice([-1, 1], size=(n, 1))
        _, m = clusters(tstat(d * flips))
        null_max[k] = np.max(np.abs(m)) if m else 0.0
    return [(s, e, m, (np.sum(null_max >= abs(m)) + 1) / (n_perm + 1))
            for (s, e), m in zip(obs_clusters, obs_mass)]
