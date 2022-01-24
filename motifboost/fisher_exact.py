import numba
import numpy as np
from numpy import abs, exp, log, sum
from scipy.special import beta


@numba.njit("f8(i8,i8)", fastmath=True)
def betaln(a, b):
    return log(abs(beta(float(a), float(b))))


@numba.njit("f8[:](i8[:],i8,i8,i8)", fastmath=True)
def logpmf(k, M, n, N):
    tot, good = M, n
    bad = tot - good
    result = np.zeros((len(k)))
    for idx, k2 in enumerate(k):
        result[idx] = (
            betaln(good + 1, 1)
            + betaln(bad + 1, 1)
            + betaln(tot - N + 1, N + 1)
            - betaln(k2 + 1, good - k2 + 1)
            - betaln(N - k2 + 1, bad - N + k2 + 1)
            - betaln(tot + 1, 1)
        )
    return result


@numba.njit("f8(i8,i8,i8,i8)", fastmath=True)
def cdf(k, M, n, N):
    if (k + 0.5) * (M + 0.5) > (n - 0.5) * (N - 0.5):
        # Less terms to sum if we calculate log(1-sf)
        k2 = np.arange(k + 1, n + 1)
        return 1 - sum(exp(logpmf(k2, M, n, N)))
    else:
        # Integration over probability mass function using logsumexp
        k2 = np.arange(0, k + 1)
        return sum(exp(logpmf(k2, M, n, N)))


# https://github.com/scipy/scipy/blob/v1.6.3/scipy/stats/stats.py#L3965-L4124
@numba.njit("f8(i8,i8,i8,i8)", fastmath=True)
def fisher_exact(a: int, b: int, c: int, d: int):
    ab = a + b
    cd = c + d
    ac = a + c
    bd = b + d
    if ab == 0 or ac == 0 or bd == 0 or cd == 0:
        return 1.0
    return min(cdf(b, ab + cd, ab, bd), 1.0)


def test_fast_fisher():
    from scipy.stats import fisher_exact as scipy_fisher_exact
    from tqdm import tqdm

    for i in tqdm(range(50)):
        for j in range(50):
            for k in range(50):
                for l in range(50):
                    scipy_p = scipy_fisher_exact([[i, j], [k, l]], "greater")[1]
                    my_p = fisher_exact(i, j, k, l)
                    if abs(my_p - scipy_p) > 0.0000001:
                        print("different", my_p, scipy_p, i, j, k, l)

    for i in tqdm(range(190, 210)):
        for j in range(190, 210):
            for k in range(190, 210):
                for l in range(190, 210):
                    scipy_p = scipy_fisher_exact([[i, j], [k, l]], "greater")[1]
                    my_p = fisher_exact(i, j, k, l)
                    if abs(my_p - scipy_p) > 0.0000001:
                        print("different", my_p, scipy_p, i, j, k, l)
