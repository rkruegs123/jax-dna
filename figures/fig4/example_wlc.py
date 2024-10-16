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
plt.rcParams.update({'font.size': 64})


output_dir = Path("figures/fig4/output")


x_init = jnp.array([39.87, 50.60, 44.54]) # initialize to the true values
x_init_si = jnp.array([x_init[0] * utils.nm_per_oxdna_length,
                       x_init[1] * utils.nm_per_oxdna_length,
                       x_init[2] * utils.oxdna_force_to_pn])

t_kelvin = utils.DEFAULT_TEMP
kT = utils.get_kt(t_kelvin)

def coth(x):
    # return 1 / jnp.tanh(x)
    return (jnp.exp(2*x) + 1) / (jnp.exp(2*x) - 1)

def calculate_x(force, l0, lps, k, kT):
    y = ((force * l0**2)/(lps*kT))**(1/2)
    x = l0 * (1 + force/k - kT/(2*force*l0) * (1 + y*coth(y)))
    return x



# forces = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.75]
forces = [0.1, 0.225, 0.55]
tom_extensions = [calculate_x(f, x_init[0], x_init[1], x_init[2], kT) for f in forces]

ex_forces = onp.linspace(0.05, 0.75, 100)
ex_extensions = [calculate_x(f, x_init[0], x_init[1], x_init[2], kT) for f in ex_forces]

fig, ax = plt.subplots(figsize=(12, 10))
# plt.plot(forces, tom_extensions)

ax.plot(ex_extensions, ex_forces, color="black", linewidth=5)
ax.scatter(tom_extensions, forces, color="black", s=500)

x_vals = tom_extensions
ax.set_xticks(x_vals)
ax.set_xticklabels([r'$\langle O \rangle ^1$', r'$\langle O \rangle ^2$', r'$\langle O \rangle ^3$'])

y_vals = forces
ax.set_yticks(forces)
ax.set_yticklabels([r'$y^1$', r'$y^2$', r'$y^3$'])

ax.tick_params(width=3, size=10)

plt.tight_layout()
# plt.show()
plt.savefig(output_dir / f"example_wlc.pdf")
