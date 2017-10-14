import nengo
import numpy as np


def gen_feed_func(vocab, vocab_items, t_present: float):

    def f(t):
        index = int(t / t_present)
        index = index % len(vocab_items)
        return vocab.parse(vocab_items[index]).v

    return f


def gen_vecs_aaron(n_items: int, d: int, simi: float):
    sphere = nengo.dists.UniformHypersphere(surface=True)
    vecs = []

    for _ in range(n_items):
        u = sphere.sample(1, d=d)  # given vector

        p = sphere.sample(1, d=d)
        q = p - u.dot(p.T) * u
        v = simi*u + np.sqrt((1 - simi**2) / q.dot(q.T))*q
        vecs.append(v.squeeze())

    return vecs


def gen_vecs_jan(n_items: int, d: int, simi: float):
    indices = np.arange(n_items, dtype=int)
    target_cor = np.ones((n_items, n_items)) * simi
    np.fill_diagonal(target_cor, 1.)

    u, s, _ = np.linalg.svd(target_cor)
    assert np.allclose(np.linalg.norm(u, axis=1), 1.)

    base_pos_vec = np.dot(u, np.diag(np.sqrt(s)))
    assert np.allclose(np.linalg.norm(base_pos_vec, axis=1), 1)

    return np.dot(
        np.random.permutation(np.eye(n_items)[:, :n_items]), base_pos_vec.T).T
