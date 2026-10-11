# From Appendix B of 'Eyes Tell More Than Mouths: The Science Companion'.
# Expects a data frame `d` in long format; see the comments in the book for the column definitions.
# Not run in the book's build (R was not available); standard lme4/mgcv usage.

# Trial-level mixed-effects model (columns: subject, item, condition, baseline, dilation)

library(lme4)
library(lmerTest)   # adds p-values for fixed effects

d$condition  <- factor(d$condition, levels = c("easy", "hard"))
contrasts(d$condition) <- c(-0.5, 0.5)      # effect coding
d$baseline_c <- d$baseline - mean(d$baseline)  # centered covariate

m <- lmer(dilation ~ condition + baseline_c +
            (1 + condition | subject) + (1 | item),
          data = d)
summary(m)
