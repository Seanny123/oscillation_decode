import nengo
import nengo.spa as spa

import numpy as np

from data_feed import BasicDataFeed
from utils import gen_vecs

import itertools

n_items = 4
t_present = 2.0
pause = 0.01
D = 16
dt = 0.001
similarity = 0.9

recog_neurons = 500

vocab = spa.Vocabulary(D)

stim_vecs = gen_vecs(n_items, D, similarity)
stim_lbl = []

for v_i, vec in enumerate(stim_vecs):
    lbl = "S%d" % v_i
    stim_lbl.append(lbl)
    vocab.add(lbl, vec)


def vocab_feed(idx, t):
    return vocab.parse(stim_lbl[idx]).v


df = BasicDataFeed(vocab_feed, np.eye(n_items), t_present, D, n_items, pause)

with nengo.Network() as model:
    in_nd = nengo.Node(df.feed)
    cor = nengo.Node(df.get_answer)

    recog = nengo.Ensemble(recog_neurons, D)

    nengo.Connection(in_nd, recog)

    p_in = nengo.Probe(in_nd, synapse=None)
    p_out = nengo.Probe(recog, synapse=0.01)
    p_cor = nengo.Probe(cor)

with nengo.Simulator(model) as sim:
    sim.run(t_present*n_items)
