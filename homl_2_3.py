# Goal: Identify the probability that sampling 1,000 people (of whom 51.1% are female)
# would result in more than 53.5% females
# The book says the chance is 10.7%. Can I find this myself?

# Ideally, sample of 1000 should yield 511 females (and 489 males).
# What is probability of sampling more than 535 females?
# This would be an excess of (535-511=) 24, representing (24/1000=) 2.4% from the mean.

# To find the answer, use a binomial distribution.

# Pseudocode: pbinom(prob = .511, n = 1000, x = 535, upper = true)
# Given the prob of female, the sample size, the x data point of interest, and indicating values more extreme

from scipy.stats import binom

n = 1000
p = .511
k1 = 485
k2 = 535

p_k1_or_less = binom.cdf(k1-1,n,p) #
p_k2_or_more = 1 - (binom.cdf(k2+1,n,p))

print(f"The probability of observing {k1} or less females out of {n} trials is: {p_k1_or_less:.3f}")
print(f"The probability of observing {k2} or more females out of {n} trials is: {p_k2_or_more:.3f}")
print(f"The probability of observing either is: {p_k2_or_more+p_k2_or_more:.3f}")
