# pupilkit — companion code

Code from **Eyes Tell More Than Mouths: The Science Companion** by James Williams (Forever Young, 2026), Appendices A and B.

A small, dependency-light toolkit for teaching pupillometry: every step is explicit so you can read it, change it and test it. It is written for learning, not as a validated clinical or research package.

## Files

| File | What it does | In the book |
|---|---|---|
| `pupilkit.py` | Blink detection, interpolation of short gaps (long gaps stay missing), low-pass filtering, downsampling, epoching, baseline correction, epoch rejection, light-reflex parameters, pupillary unrest index, pupil response function and deconvolution, cluster-based permutation test | Appendix A |
| `demo.py` | Simulates 20 participants with easy and hard trials, blinks, drift, noise and a 2-second loss of tracking, then runs the full pipeline and the analyses | Appendix A, worked example |
| `power_sim.py` | Power by simulation for participants and trials | Appendix B |
| `R/mixed_model_lme4.R` | Trial-level mixed-effects model (lme4) | Appendix B |
| `R/gamm_ar1_mgcv.R` | Generalized additive mixed model with AR(1) errors (mgcv) | Appendix B |
| `expected_output_*.txt` | What the Python scripts print, for checking your setup | |

## Quick start

Python 3.9 or later.

```
pip install -r requirements.txt
python demo.py
python power_sim.py
```

Compare what you see with `expected_output_demo.txt` and `expected_output_power_sim.txt`. The scripts use fixed random seeds, so the numbers should match.

The R scripts need `lme4`, `lmerTest`, `mgcv` and `itsadug`, and a data frame `d` in long format with the columns named at the top of each file.

## Notes

- `downsample` requires the new rate to divide the old one exactly (250 to 50 Hz works; 250 to 60 Hz raises an error instead of silently giving 62.5 Hz).
- In the demo, trials are 6 s apart, so responses do not overlap; the deconvolution result there is not a test of separating overlapping responses.
- The cluster test tells you whether conditions differ; the extent of a cluster is descriptive and is not an estimate of when an effect starts or ends.

## License

MIT (see `LICENSE`). If you use this code in teaching or research, a citation of the book is appreciated:

Williams, J. (2026). *Eyes Tell More Than Mouths: The Science Companion*. Forever Young.
