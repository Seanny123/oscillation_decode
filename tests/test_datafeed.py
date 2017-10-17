from data_feed import BasicDataFeed
from constants import dt

import numpy as np
import matplotlib.pyplot as plt


dims = 4
t_len = 0.1
pause = 0.01
n_items = 4

cor = np.eye(n_items)


def dataset_func(idx, t):
    return cor[idx] * (t*10)


df = BasicDataFeed(dataset_func, np.eye(n_items), t_len, dims, n_items, pause)

t_steps = list(np.arange(0, 2*n_items*(t_len+pause), dt))
df_out = []
ans_out = []

for tt in t_steps:
    df_out.append(df.feed(tt))
    ans_out.append(df.get_answer(tt))

plt.plot(df_out)

plt.gca().set_prop_cycle(None)

plt.plot(ans_out)
plt.show()
