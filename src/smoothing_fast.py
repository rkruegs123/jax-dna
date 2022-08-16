import pdb
import numpy as np


def get_f1_smoothing_params(eps, x0, a, xc, x_low, x_high):

    def solve_f1_b(x):
        return a**2*(-np.exp(a*(3*x0 + 2*xc)) + 2*np.exp(a*(x + 2*x0 + 2*xc)) \
                     - np.exp(a*(2*x + x0 + 2*xc)))*np.exp(-2*a*x) \
                     / (2*np.exp(a*(x + 2*xc)) \
                        + np.exp(a*(2*x + x0)) \
                        - 2*np.exp(a*(2*x + xc)) \
                        - np.exp(a*(x0 + 2*xc)))
    solved_b_low = solve_f1_b(x_low)
    solved_b_high = solve_f1_b(x_high)

    def solve_f1_xc_star(x):
        return (a*x*np.exp(a*(x + 2*xc)) - a*x*np.exp(a*(x0 + 2*xc))
                + 2*np.exp(a*(x + 2*xc)) + np.exp(a*(2*x + x0)) - 2*np.exp(a*(2*x + xc))
                - np.exp(a*(x0 + 2*xc)))*np.exp(-2*a*xc)/(a*(np.exp(a*x) - np.exp(a*x0)))

    solved_xc_low = solve_f1_xc_star(x_low)
    solved_xc_high = solve_f1_xc_star(x_high)

    return solved_b_low, solved_xc_low, solved_b_high, solved_xc_high


def get_f2_smoothing_params(k, x0, xc, x_low, x_high):
    def solve_f2_b(x):
        return (x - x0)**2/(2*(x - xc)*(x - 2*x0 + xc))

    def solve_f2_xc_star(x):
        return (x*x0 - 2*x0*xc + xc**2)/(x - x0)

    solved_b_low = solve_f2_b(x_low)
    solved_b_high = solve_f2_b(x_high)

    solved_xc_low = solve_f2_xc_star(x_low)
    solved_xc_high = solve_f2_xc_star(x_high)

    return solved_b_low, solved_xc_low, solved_b_high, solved_xc_high


def get_f3_smoothing_params(r_star, eps, sigma):
    def solve_f3_b(x):
        return -36*sigma**6*(-2*sigma**6 + x**6)**2 \
            / (x**14*(-sigma + x)*(sigma + x)*(sigma**2 - sigma*x + x**2) \
               * (sigma**2 + sigma*x + x**2))

    def solve_f3_xc(x):
        return x*(-7*sigma**6 + 4*x**6)/(3*(-2*sigma**6 + x**6))

    solved_b = solve_f3_b(r_star)
    solved_xc = solve_f3_xc(r_star)

    return solved_b, solved_xc


tol = 1e-10
def get_f4_smoothing_params(a, x0, delta_x_star):
    def solve_f4_b(x):
        return -a**2*(-x + x0)**2/(a*x**2 - 2*a*x*x0 + a*x0**2 - 1)

    solved_b_plus = solve_f4_b(x0 + delta_x_star)
    solved_b_minus = solve_f4_b(x0 - delta_x_star)

    assert(np.abs(solved_b_plus - solved_b_minus) < tol) # FIXME: don't understand why this is true

    def solve_f4_xc(x):
        return (-a*x*x0 + a*x0**2 - 1)/(a*(-x + x0))

    solved_xc_plus = solve_f4_xc(x0 + delta_x_star)
    solved_delta_xc_plus = solved_xc_plus - x0

    solved_xc_minus = solve_f4_xc(x0 - delta_x_star)
    solved_delta_xc_minus = x0 - solved_xc_minus

    assert(np.abs(solved_delta_xc_plus - solved_delta_xc_minus) < tol) # FIXME: don't understand why this is true

    return solved_b_plus, solved_delta_xc_plus


def get_f5_smoothing_params(a, x_star):
    def solve_f5_b(x, x0):
        return -a**2*(x - x0)**2/(a*x**2 - 2*a*x*x0 + a*x0**2 - 1)

    solved_b = solve_f5_b(x=x_star, x0=0.0)

    def solve_f5_xc(x, x0):
        return (a*x*x0 - a*x0**2 + 1)/(a*(x - x0))

    solved_xc = solve_f5_xc(x=x_star, x0=0.0)

    return solved_b, solved_xc


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from potential import f1, f2, f3, f4, f5

    eps = 1.3448 # FIXME: + 2.6568 kT
    a = 6
    x0 = 0.4
    xc = 0.9
    x_low = 0.32
    x_high = 0.75
    b_low, xc_low, b_high, xc_high = get_f1_smoothing_params(eps, x0, a, xc, x_low, x_high)
    xs = np.linspace(0.25, 0.85, 30)
    vs = [f1(x, x_low, x_high, xc_low, xc_high, eps, a, x0, xc, b_low, b_high) for x in xs]
    plt.plot(xs, vs)
    plt.axvline(x=x_low, linestyle='--')
    plt.axvline(x=x_high, linestyle='--')
    plt.show()
