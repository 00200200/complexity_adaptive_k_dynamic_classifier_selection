import numpy as np
from scipy import stats


def cv52cft(a, b):
    d = a.reshape(2, 5) - b.reshape(2, 5)
    f = np.sum(np.power(d, 2)) / (2 * np.sum(np.var(d, axis=0, ddof=0)))
    p = 1 - stats.f.cdf(f, 10, 5)
    return f, p


def t_test_corrected(a, b, J=2, k=5):
    if J * k != a.shape[0]:
        raise Exception(
            "%i scores received, but J=%i, k=%i (J*k=%i)" % (a.shape[0], J, k, J * k)
        )

    d = a - b
    bar_d = np.mean(d)

    bar_sigma_2 = np.var(d.reshape(-1), ddof=1)
    bar_sigma_2_mod = (1 / (J * k) + 1 / (k - 1)) * bar_sigma_2
    t_stat = bar_d / np.sqrt(bar_sigma_2_mod)
    pval = stats.t.sf(np.abs(t_stat), (k * J) - 1) * 2
    return t_stat, pval
