# From Appendix B of 'Eyes Tell More Than Mouths: The Science Companion'.
# Expects a data frame `d` in long format; see the comments in the book for the column definitions.
# Not run in the book's build (R was not available); standard lme4/mgcv usage.

# Generalized additive mixed model with AR(1) errors (columns: subject, condition, time, pupil, start_event)

library(mgcv)
library(itsadug)    # plotting and AR1 helpers

d$subject   <- factor(d$subject)
d$condition <- factor(d$condition)

# 1. Fit without autocorrelation to estimate it
m0 <- bam(pupil ~ condition + s(time, by = condition, k = 20) +
            s(time, subject, bs = "fs", m = 1, k = 10),
          data = d, method = "fREML", discrete = TRUE)
rho <- acf(resid(m0), plot = FALSE)$acf[2]

# 2. Refit with an AR(1) error model; start_event is TRUE
#    at the first sample of each trial
m1 <- bam(pupil ~ condition + s(time, by = condition, k = 20) +
            s(time, subject, bs = "fs", m = 1, k = 10),
          data = d, method = "fREML", discrete = TRUE,
          rho = rho, AR.start = d$start_event)
summary(m1)

# 3. Plot the estimated difference curve with confidence band
plot_diff(m1, view = "time",
          comp = list(condition = c("hard", "easy")), rm.ranef = TRUE)
