import numpy as np
from scipy import stats


# Alpaydin test ?
def cv52cft(a, b):
    d = a.reshape(2, 5) - b.reshape(2, 5)
    f = np.sum(np.power(d, 2)) / (2 * np.sum(np.var(d, axis=0, ddof=0)))
    p = 1 - stats.f.cdf(f, 10, 5)
    return f, p
