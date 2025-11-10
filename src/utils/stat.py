import numpy as np
from scipy import stats


def t_test_corrected(a, b, J=2, k=5):
    if J * k != a.shape[0]:
        raise ValueError(
            f"{a.shape[0]} scores received, but J={J}, k={k} (J*k={J * k})"
        )

    d = a - b
    bar_d = np.mean(d)

    bar_sigma_2 = np.var(d.reshape(-1), ddof=1)
    bar_sigma_2_mod = (1 / (J * k) + 1 / (k - 1)) * bar_sigma_2
    t_stat = bar_d / np.sqrt(bar_sigma_2_mod)
    pval = stats.t.sf(np.abs(t_stat), (k * J) - 1) * 2
    return t_stat, pval
