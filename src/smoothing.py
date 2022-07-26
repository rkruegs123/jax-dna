from sympy import *
from sympy.solvers import solve

import pdb
from copy import deepcopy

# setup
x, a, x0, b, xc, eps, sigma, k = symbols('x, a, x0, b, xc, eps, sigma, k')
xc_star = symbols('xc_star') # used in f1 and f2, where rc is given but rc_{high/low} is not ("star" = "high" or "low")
vsmooth_val = b*(xc - x)**2
vsmooth_dx = diff(vsmooth_val, x)


# f1
eps_vsmooth_val_1 = eps * b * (xc_star - x)**2
eps_vsmooth_dx_1 = diff(eps_vsmooth_val_1, x)
vmorse_val1 = eps * (1 - exp(-a * (x - x0)))**2 # x, eps, x0, a
vmorse_val2 = eps * (1 - exp(-a * (xc - x0)))**2 # xc, eps, x0, a
vmorse_sub_vmorse_val = vmorse_val1 - vmorse_val2
vmorse_sub_vmorse_dx = diff(vmorse_sub_vmorse_val, x)
f1_solved = solve([vmorse_sub_vmorse_val - eps_vsmooth_val_1,
                   vmorse_sub_vmorse_dx - eps_vsmooth_dx_1],
                  (b, xc_star), dict=True)
f1_solved_b = f1_solved[0][b]
f1_solved_xc_star = f1_solved[0][xc_star]
def get_f1_smoothing_params(eps, x0, a, xc, x_low, x_high):
    # Should be continuous at x_low and x_high
    solved_b_low = f1_solved_b.subs({'x': x_low, 'eps': eps, 'x0': x0, 'a': a})
    solved_xc_star_low = f1_solved_xc_star.subs({'x': x_low, 'eps': eps, 'x0': x0, 'a': a})

    solved_b_high = f1_solved_b.subs({'x': x_high, 'eps': eps, 'x0': x0, 'a': a})
    solved_xc_star_high = f1_solved_xc_star.subs({'x': x_high, 'eps': eps, 'x0': x0, 'a': a})

    return float(solved_b_low.evalf()), float(solved_xc_star_low.evalf()), float(solved_b_high.evalf()), float(solved_xc_star_high.evalf())


# f2
vharm_val1 = k / 2 * (x - x0)**2 # x, k, x0
vharm_val2 = k / 2 * (xc - x0)**2 # xc, k, x0
vharm_sub_vharm_val = vharm_val1 - vharm_val2
vharm_sub_vharm_dx = diff(vharm_sub_vharm_val, x)
k_vsmooth_val = k * b * (xc_star - x)**2 # e.g. r, b_low, rc_low
k_vsmooth_dx = diff(k_vsmooth_val, x)
f2_solved = solve([vharm_sub_vharm_val - k_vsmooth_val,
                   vharm_sub_vharm_dx - k_vsmooth_dx],
                  (b, xc_star), dict=True) # note we solve for xc_star here rather than for xc
f2_solved_b = f2_solved[0][b]
f2_solved_xc_star = f2_solved[0][xc_star]
def get_f2_smoothing_params(k, x0, xc, x_low, x_high):
    # Should be continuous at x_low and x_high
    solved_b_low = f2_solved_b.subs({'x': x_low, 'k': k, 'x0': x0, 'xc': xc})
    solved_xc_low = f2_solved_xc_star.subs({'x': x_low, 'k': k, 'x0': x0, 'xc': xc})

    solved_b_high = f2_solved_b.subs({'x': x_high, 'k': k, 'x0': x0, 'xc': xc})
    solved_xc_high = f2_solved_xc_star.subs({'x': x_high, 'k': k, 'x0': x0, 'xc': xc})
    return float(solved_b_low.evalf()), float(solved_xc_low.evalf()), float(solved_b_high.evalf()), float(solved_xc_high.evalf())


# f3
vlj_val = 4 * eps * ((sigma / x)**12 - (sigma / x)**6)
vlj_dx = diff(vlj_val, x)
eps_vsmooth_val_2 = eps * vsmooth_val
eps_vsmooth_dx_2 = diff(eps_vsmooth_val_2, x)
f3_solved = solve([vlj_val - eps_vsmooth_val_2, vlj_dx - eps_vsmooth_dx_2], (b, xc), dict=True)
f3_solved_b = f3_solved[0][b]
f3_solved_xc = f3_solved[0][xc]
def get_f3_smoothing_params(r_star, eps, sigma):
    # Must be continuous at r_star
    solved_b = f3_solved_b.subs({'x': r_star, 'sigma': sigma, 'eps': eps})
    solved_xc = f3_solved_xc.subs({'x': r_star, 'sigma': sigma, 'eps': eps})

    return float(solved_b.evalf()), float(solved_xc.evalf())


# f4
vmod_val = 1 - a*(x - x0)**2
vmod_dx = diff(vmod_val, x)
f4_solved = solve([vmod_val - vsmooth_val, vmod_dx - vsmooth_dx], (b, xc), dict=True)
# FIXME: check that values are in it
f4_solved_b = f4_solved[0][b]
f4_solved_xc = f4_solved[0][xc]
def get_f4_smoothing_params(a, x0, delta_x_star):
    # Must be continuous at x_0 +- delta_x_star
    solved_b_plus = f4_solved_b.subs({'x': x0 + delta_x_star, 'a': a, 'x0': x0, 'x0': x0})
    solved_b_minus = f4_solved_b.subs({'x': x0 - delta_x_star, 'a': a, 'x0': x0, 'x0': x0})

    assert(solved_b_plus == solved_b_minus) # FIXME: don't understand why this is true

    # Solve total xc's, then adjust
    solved_xc_plus = f4_solved_xc.subs({'x': x0 + delta_x_star, 'a': a, 'x0': x0, 'x0': x0})
    solved_delta_xc_plus = solved_xc_plus - x0
    solved_xc_minus = f4_solved_xc.subs({'x': x0 - delta_x_star, 'a': a, 'x0': x0, 'x0': x0})
    solved_delta_xc_minus = x0 - solved_xc_minus

    assert(solved_delta_xc_plus == solved_delta_xc_minus) # FIXME: don't understand why this is true

    return float(solved_b_plus.evalf()), float(solved_delta_xc_plus.evalf())


# f5
f5_solved = deepcopy(f4_solved)
f5_solved_b = f5_solved[0][b]
f5_solved_xc = f5_solved[0][xc]
def get_f5_smoothing_params(a, x_star):
    solved_b = f5_solved_b.subs({'x': x_star, 'a': a, 'x0': 0.0})
    solved_xc = f5_solved_b.subs({'x': x_star, 'a': a, 'x0': 0.0})
    return float(solved_b.evalf()), float(solved_xc.evalf())


"""
TODO:
- load toml in such a way that we can override existing values
- check smoothing parameters via plotting
- make a "get_params" in a `utils.py` that takes in a .toml, calls these, subs with existing, and upates the dictionary
- then, using this, can compare the if statements with jnp.where. Also, with the multiplicative isotropic cutoff
- Note: tomorrow HAS to be Liu stuff AND Max stuff!
"""



if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from potential import f1, f2, f3, f4, f5
    import numpy as np



    # Test cross-stacking f2
    k = 47.5
    x0 = 0.575
    xc = 0.675
    x_low = 0.495
    x_high = 0.655
    b_low, xc_low, b_high, xc_high = get_f2_smoothing_params(k, x0, xc, x_low, x_high)

    xs = np.linspace(0.3, 0.8, 100)
    vs = [f2(x, x_low, x_high, xc_low, xc_high, k, x0, xc, b_low, b_high) for x in xs]
    plt.plot(xs, vs)
    plt.axvline(x=x_low, linestyle='--')
    plt.axvline(x=x_high, linestyle='--')
    plt.show()


    # Test excluded volume f3
    """
    r_star = 0.675
    eps = 2.0
    sigma = 0.70
    b, rc = get_f3_smoothing_params(r_star, eps, sigma)

    xs = np.linspace(0.625, 0.75, 30)
    vs = [f3(x, r_star, rc, eps, sigma, b) for x in xs]
    # vs = f3(xs, r_star, rc, eps, sigma, b)
    plt.plot(xs, vs)
    plt.axvline(x=r_star, linestyle='--')
    plt.axvline(x=rc, linestyle='--')
    plt.show()
    """


    # Test stacking f4_theta4
    """
    a = 1.30
    x0 = 0.0
    delta_x_star = 0.8
    b, delta_xc = get_f4_smoothing_params(a, x0, delta_x_star)

    ## Check entire potential
    xs = np.linspace(-1.5, 1.5, 40)
    vs = [f4(x, x0, delta_x_star, delta_xc, a, b) for x in xs]
    plt.plot(xs, vs)
    plt.axvline(x0 - delta_xc, linestyle='--')
    plt.axvline(x0 + delta_xc, linestyle='--')
    plt.show()

    ## Check left boundary
    xs = np.linspace(-0.975, -0.95, 30)
    vs = [f4(x, x0, delta_x_star, delta_xc, a, b) for x in xs]
    plt.plot(xs, vs)
    plt.axvline(x0 - delta_xc, linestyle='--')
    plt.show()

    ## Check right boundary
    xs = np.linspace(0.95, 0.975, 30)
    vs = [f4(x, x0, delta_x_star, delta_xc, a, b) for x in xs]
    plt.plot(xs, vs)
    plt.axvline(x0 + delta_xc, linestyle='--')
    plt.show()
    """
