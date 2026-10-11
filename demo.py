"""demo.py -- simulate a small pupil experiment and analyze it."""
import numpy as np
from scipy import stats
import pupilkit as pk

rng = np.random.default_rng(42)
FS = 250                      # sampling rate (Hz)
N_SUBJ, N_TRIALS = 20, 30     # participants, trials per condition
ITI = 6.0                     # seconds between trial onsets

def simulate_participant(effect_mm=0.10):
    """One continuous recording with 'easy' and 'hard' trials."""
    conds = rng.permutation(["easy"] * N_TRIALS + ["hard"] * N_TRIALS)
    dur = len(conds) * ITI + 5
    n = int(dur * FS)
    t = np.arange(n) / FS
    h = pk.pupil_response_function(FS)
    stick = {c: np.zeros(n) for c in ("easy", "hard")}
    onsets = {"easy": [], "hard": []}
    for k, c in enumerate(conds):
        on = 2.0 + k * ITI
        onsets[c].append(on)
        stick[c][int(on * FS)] = 1.0
    eff = rng.normal(effect_mm, 0.08)                   # person-specific effect
    amp = {"easy": 0.15, "hard": 0.15 + eff}
    for c in stick:                                      # trial-to-trial jitter
        stick[c] *= amp[c] + rng.normal(0, 0.15, n) * (stick[c] > 0)
    resp = sum(np.convolve(stick[c], h)[:n] for c in stick)
    drift = 0.15 * np.sin(2 * np.pi * t / 90)          # slow drift
    noise = np.convolve(rng.normal(0, 0.25, n), np.ones(25) / 25, "same")
    pupil = 4.0 + rng.normal(0, 0.3) + drift + resp + noise
    # insert blinks: ~one every 4 s, 150 ms long, with distorted edges
    for b in rng.uniform(0, dur - 1, int(dur / 4)):
        i = int(b * FS)
        pupil[i : i + int(0.15 * FS)] = 0.0
        pupil[max(i - 10, 0) : i] -= 0.4
    loss = int(rng.uniform(10, dur - 10) * FS)          # one 2-s loss of tracking
    pupil[loss : loss + 2 * FS] = 0.0
    return pupil, onsets

subj_easy, subj_hard, deconv, n_tot, n_keep = [], [], [], 0, 0
for s in range(N_SUBJ):
    raw, onsets = simulate_participant()
    bad = pk.detect_blinks(raw, FS, pad_ms=100)
    clean = pk.interpolate_missing(raw, bad, max_gap_s=1.0, fs=FS)
    clean = pk.lowpass(clean, FS, 4.0)                  # long gaps stay NaN
    x50 = pk.downsample(clean, FS, 50)                  # 50 Hz
    times, ep_e = pk.epoch(x50, 50, onsets["easy"], -0.5, 3.0)
    _, ep_h = pk.epoch(x50, 50, onsets["hard"], -0.5, 3.0)
    n_tot += len(ep_e) + len(ep_h)
    ep_e = pk.baseline_correct(pk.reject_epochs(ep_e, times), times)
    ep_h = pk.baseline_correct(pk.reject_epochs(ep_h, times), times)
    n_keep += len(ep_e) + len(ep_h)                     # trials actually analyzed
    subj_easy.append(np.nanmean(ep_e, 0)); subj_hard.append(np.nanmean(ep_h, 0))
    deconv.append(pk.deconvolve(x50 - np.nanmean(x50), 50, onsets))

A, B = np.array(subj_hard), np.array(subj_easy)
print(f"Trials analyzed: {n_keep} of {n_tot} (excluded {n_tot - n_keep})")

# (a) time-window analysis: mean dilation 1.0-2.0 s after onset
win = (times >= 1.0) & (times < 2.0)
t_val, p_val = stats.ttest_rel(A[:, win].mean(1), B[:, win].mean(1))
dz = (A[:, win].mean(1) - B[:, win].mean(1))
dz = dz.mean() / dz.std(ddof=1)
print(f"Window 1-2 s: t({N_SUBJ-1}) = {t_val:.2f}, p = {p_val:.4f}, dz = {dz:.2f}")
# (b) cluster-based permutation test over the whole epoch
for s_i, e_i, mass, p in pk.cluster_permutation_test(A, B, n_perm=1000):
    print(f"Cluster {times[s_i]:.2f}-{times[e_i]:.2f} s, mass = {mass:.1f}, p = {p:.4f}")
# (c) deconvolution: recovered amplitudes per condition
hard = np.array([d["hard"] for d in deconv]); easy = np.array([d["easy"] for d in deconv])
print(f"Deconvolved amplitude: easy {easy.mean():.3f}, hard {hard.mean():.3f} (simulated 0.150 / 0.250 on average)")
# (d) light-reflex parameters for a simulated flash response
fs = 250
tt = np.arange(-0.5, 4, 1 / fs)
s = np.clip(tt - 0.23, 0, None)
plr = 6.0 - 2.0 * (1 - np.exp(-s / 0.28)) ** 2 * np.exp(-np.clip(s - 0.6, 0, None) / 1.4)
print({k: round(float(v), 3) for k, v in pk.plr_parameters(plr, fs).items()})
# (e) pupillary unrest index for 2 minutes of simulated data
x = 6 + np.cumsum(rng.normal(0, 0.003, 120 * 25))
print(f"PUI = {pk.pupillary_unrest_index(x, 25):.2f} mm/min")
