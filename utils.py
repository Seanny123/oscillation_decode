import nengo
import numpy as np

from nengolib.stats import sphere as depedent_sphere


def gen_feed_func(vocab, vocab_items, t_present: float):

    def f(t):
        index = int(t / t_present)
        index = index % len(vocab_items)
        return vocab.parse(vocab_items[index]).v

    return f


def graham_m(d: int, simi: float) -> np.array:

    def proj(u: np.array, v: np.array) -> np.array:
        return u.dot(v.T) / u.dot(u.T) * u

    uniform_sphere = nengo.dists.UniformHypersphere(surface=True)
    # output matrix
    M = np.zeros((d, d))
    # Gram-Schmidt orthogonalization of S
    Q = np.zeros((d, d))
    # some random samples used to form Q
    S = uniform_sphere.sample(d, d=d)

    # Simultaneously apply Gram-Schmidt to S to compute Q while
    # using Q to form M. This exploits the fact that
    # Q[i, :].T.dot(M[j, :]) == 0 for all j < i.
    for i in range(d):
        Q[i, :] = S[i, :]
        a = simi / ((i - 1) * simi + 1)
        for j in range(i):
            Q[i, :] -= proj(Q[j, :], S[i, :])
            M[i, :] += a * M[j, :]
        M[i, :] += np.sqrt((1 - i * a * simi) / Q[i, :].T.dot(Q[i, :])) * Q[i, :]

    return M


def gen_vecs(n_items: int, d: int, simi: float) -> np.array:
    M = graham_m(d, simi)
    return M[:n_items]


def gen_vecs_iter(n_items: int, d: int, simi: float) -> np.array:

    M = graham_m(d, simi)

    good = []
    bad = []
    for p in depedent_sphere.sample(int(2e4), d=d):
        if np.all(M.dot(p.T) >= simi):
            good.append(p)
            if len(good) > n_items:
                break
        else:
            bad.append(p)

    assert len(good) > 0
    good = np.asarray(good)
    assert np.all(good.dot(good.T) >= simi)

    return good
