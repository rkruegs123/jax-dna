from sympy import *
from sympy.solvers import solve

import pdb


# setup
x, a, x0, b, xc, eps, sigma = symbols('x, a, x0, b, xc, eps, sigma')
vsmooth_val = b*(xc - x)**2
vsmooth_dx = diff(vsmooth_val, x)

# Vmod and Vsmooth
vmod_val = 1 - a*(x - x0)**2
vmod_dx = diff(vmod_val, x)
mod_smooth_solved = solve([vmod_val - vsmooth_val, vmod_dx - vsmooth_dx], (b, xc), dict=True)
# FIXME: check that values are in it
print(mod_smooth_solved)


def get_f4_params(a, x0, b, delta_x_star):
    raise NotImplementedError


# f3
vlj_val = 4 * eps * ((sigma / x)**12 - (sigma / x)**6)
vlj_dx = diff(vlj_val, x)
eps_vsmooth_val = eps * vsmooth_val
eps_vsmooth_dx = diff(eps_vsmooth_val, x)
def get_f3_smoothing_params(r_star, eps, sigma):
    pass



"""
TODO:
- load toml in such a way that we can override existing values
- check smoothing parameters via plotting
- make a "get_params" in a `utils.py` that takes in a .toml, calls these, subs with existing, and upates the dictionary
- then, using this, can compare the if statements with jnp.where. Also, with the multiplicative isotropic cutoff
- Note: tomorrow HAS to be Liu stuff AND Max stuff!
"""
