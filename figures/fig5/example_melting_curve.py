import pdb
from pathlib import Path
from tqdm import tqdm
import time
import numpy as onp
from copy import deepcopy
import functools
import matplotlib.pyplot as plt
import shutil
import argparse
from matplotlib import rc

import jax.numpy as jnp
from jax import jit, vmap

from jax_dna.common import utils


rc('text', usetex=True)
plt.rcParams.update({'font.size': 48})


output_dir = Path("figures/fig5/output")

temps = jnp.array([320.00, 321.58, 323.16, 324.74, 326.32, 327.89, 329.47, 331.05, 332.63, 334.21, 335.79, 337.37, 338.95, 340.53, 342.11, 343.68, 345.26, 346.84, 348.42, 350.00])
finfs = jnp.array([0.989013, 0.982473, 0.972149, 0.956016, 0.931200, 0.893935, 0.839955, 0.765704, 0.670517, 0.558935, 0.441065, 0.329483, 0.234296, 0.160045, 0.106065, 0.068800, 0.043984, 0.027851, 0.017527, 0.010987])


x = jnp.flip(finfs)
y = jnp.flip(temps)
xin = jnp.arange(0.01, 1., 0.01)
interp_vals = jnp.interp(xin, x, y)
tm = interp_vals[49]


fig, ax = plt.subplots(figsize=(12, 10))
ax.plot(jnp.flip(interp_vals), jnp.flip(xin), color="black", linewidth=5)
ax.axvline(x=tm, linestyle="--", color="red", linewidth=3)
ax.axhline(y=0.5, linestyle="--", color="red", linewidth=3)

ax.set_ylabel("Duplex Yield")
ax.set_xlabel("T / K")

ax.set_yticks([0.0, 0.5, 1.0])
ax.set_yticklabels([r'$0.0$', r'$0.5$', r'$1.0$'])

ax.set_xticks([320, 330, tm, 340, 350])
ax.set_xticklabels([r'$320$', r'$330$', r'$T_m$', r'$340$', r'$350$'])

ax.tick_params(width=3, size=10)

plt.tight_layout()

# plt.show()
plt.savefig(output_dir / "ex_melting_curve.pdf")
