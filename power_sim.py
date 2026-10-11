"""power_sim.py -- estimate statistical power by simulation."""
import numpy as np
from scipy import stats

def power_paired(n, dz, alpha=0.05, n_sim=5000, seed=1):
    """Power of a paired t-test for standardized effect dz with n people."""
    rng = np.random.default_rng(seed)
    diffs = rng.normal(dz, 1.0, size=(n_sim, n))
    t = diffs.mean(1) / (diffs.std(1, ddof=1) / np.sqrt(n))
    p = 2 * stats.t.sf(np.abs(t), n - 1)
    return np.mean(p < alpha)

def power_trials(n_subj, n_trials, effect, sd_subj, sd_trial,
                 alpha=0.05, n_sim=2000, seed=2):
    """Power when the effect is estimated from noisy single trials.

    effect   : true mean difference (mm) between conditions
    sd_subj  : between-person SD of the true effect (mm)
    sd_trial : SD of single-trial responses (mm)
    """
    rng = np.random.default_rng(seed)
    hits = 0
    for _ in range(n_sim):
        true = rng.normal(effect, sd_subj, n_subj)
        # mean of n_trials per condition; difference of two condition means
        se = sd_trial * np.sqrt(2.0 / n_trials)
        observed = true + rng.normal(0, se, n_subj)
        t, p = stats.ttest_1samp(observed, 0.0)
        hits += p < alpha
    return hits / n_sim

if __name__ == "__main__":
    for dz in (0.8, 0.5, 0.3):
        print(f"dz = {dz}: n = 34 gives power {power_paired(34, dz):.2f}")
    for k in (10, 30, 90):
        pw = power_trials(30, k, effect=0.05, sd_subj=0.05, sd_trial=0.25)
        print(f"30 people, {k:>2} trials per condition: power {pw:.2f}")
