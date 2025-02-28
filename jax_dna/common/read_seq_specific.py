# ruff: noqa
# fmt: off
import pdb
from pathlib import Path
import numpy as onp

import jax.numpy as jnp

from jax_dna.common.utils import DNA_ALPHA, get_kt, DEFAULT_TEMP


STCK_UNCOUPLED_PAIRS_OXDNA1 = ["GC", "CG", "AT", "TA"]
STCK_COUPLED_PAIRS_OXDNA1 = [("GG", "CC"), ("GA", "TC"), ("AG", "CT"),
                             ("TG", "CA"), ("GT", "AC"), ("AA", "TT")]
STCK_UNCOUPLED_PAIRS_OXDNA2 = ["GC", "CG", "AT", "TA", "AA", "TT"]
STCK_COUPLED_PAIRS_OXDNA2 = [("GG", "CC"), ("GA", "TC"), ("AG", "CT"),
                             ("TG", "CA"), ("GT", "AC")]

def constrain(hb_mult, stack_mult, coupled_pairs=STCK_COUPLED_PAIRS_OXDNA1):

    stck_coupled_pairs_pair1_idxs = jnp.array([(DNA_ALPHA.index(nt1), DNA_ALPHA.index(nt2)) for (nt1, nt2), _ in coupled_pairs])
    stck_coupled_pairs_pair2_idxs = jnp.array([(DNA_ALPHA.index(nt1), DNA_ALPHA.index(nt2)) for _, (nt1, nt2) in coupled_pairs])

    # Constrain hydrogen bonding
    hb_ta_mult = hb_mult[DNA_ALPHA.index("T"), DNA_ALPHA.index("A")]
    hb_gc_mult = hb_mult[DNA_ALPHA.index("G"), DNA_ALPHA.index("C")]

    hb_mult = jnp.zeros((4, 4))
    hb_mult = hb_mult.at[DNA_ALPHA.index("A"), DNA_ALPHA.index("T")].set(hb_ta_mult)
    hb_mult = hb_mult.at[DNA_ALPHA.index("T"), DNA_ALPHA.index("A")].set(hb_ta_mult)
    hb_mult = hb_mult.at[DNA_ALPHA.index("C"), DNA_ALPHA.index("G")].set(hb_gc_mult)
    hb_mult = hb_mult.at[DNA_ALPHA.index("G"), DNA_ALPHA.index("C")].set(hb_gc_mult)

    # Constrain stacking
    pair1_vals = stack_mult[stck_coupled_pairs_pair1_idxs[:, 0], stck_coupled_pairs_pair1_idxs[:, 1]]
    stack_mult = stack_mult.at[stck_coupled_pairs_pair2_idxs[:, 0], stck_coupled_pairs_pair2_idxs[:, 1]].set(pair1_vals)

    return hb_mult, stack_mult

def read_ss_oxdna(
        ss_path,
        default_f1_eps_hb=1.077,
        default_f1_eps_base_stack=1.3448,
        default_f1_eps_kt_coeff_stack=2.6568,
        enforce_symmetry=True,
        t_kelvin=DEFAULT_TEMP,
        coupled_pairs=STCK_COUPLED_PAIRS_OXDNA1,
        uncoupled_pairs=STCK_UNCOUPLED_PAIRS_OXDNA1,
):

    ss_path = Path(ss_path)
    assert(ss_path.exists())

    with open(ss_path, "r") as f:
        lines = f.readlines()

    vals = dict()
    for unstripped_line in lines:
        line = unstripped_line.strip()
        if not line:
            continue
        if lines[0] == "#":
            continue

        tokens = line.split()
        assert(len(tokens) == 3)
        assert(tokens[1] == "=")
        key = tokens[0]
        val = tokens[2]
        if val[-1] == "f":
            val = val[:-1]

        try:
            val = float(val)
        except:
            raise ValueError(f"Invalid float: {val}")

        vals[key] = val

    # Parse HB table
    hb_mult = onp.zeros((4, 4)) # Note: can maybe just use this table instead of the is_valid thing

    hb_at_val = None
    if "HYDR_A_T" in vals:
        hb_at_val = vals["HYDR_A_T"]
    if "HYDR_T_A" in vals:
        if hb_at_val is not None and enforce_symmetry:
            assert(hb_at_val == vals["HYDR_T_A"])
        else:
            hb_at_val = vals["HYDR_T_A"]
    assert(hb_at_val is not None)
    hb_at_mult = hb_at_val / default_f1_eps_hb

    hb_gc_val = None
    if "HYDR_G_C" in vals:
        hb_gc_val = vals["HYDR_G_C"]
    if "HYDR_C_G" in vals:
        if hb_gc_val is not None and enforce_symmetry:
            assert(hb_gc_val == vals["HYDR_C_G"])
        else:
            hb_gc_val = vals["HYDR_C_G"]
    assert(hb_gc_val is not None)
    hb_gc_mult = hb_gc_val / default_f1_eps_hb

    ## FIXME: store in a table
    hb_mult[DNA_ALPHA.index("A"), DNA_ALPHA.index("T")] = hb_at_mult
    hb_mult[DNA_ALPHA.index("T"), DNA_ALPHA.index("A")] = hb_at_mult
    hb_mult[DNA_ALPHA.index("G"), DNA_ALPHA.index("C")] = hb_gc_mult
    hb_mult[DNA_ALPHA.index("C"), DNA_ALPHA.index("G")] = hb_gc_mult

    # Parse stack table
    stack_mult = onp.zeros((4, 4))
    stck_fact_eps = vals["STCK_FACT_EPS"]

    kt = get_kt(t_kelvin) # Note: could be any value for kT within reason
    sa_f1_eps = default_f1_eps_base_stack + kt * default_f1_eps_kt_coeff_stack

    for nt1, nt2 in uncoupled_pairs:
        key_name = f"STCK_{nt1}_{nt2}"
        val = vals[key_name]
        calc_f1_eps = val * (1.0 - stck_fact_eps + (kt * 9.0 * stck_fact_eps))
        mult = calc_f1_eps / sa_f1_eps

        stack_mult[DNA_ALPHA.index(nt1), DNA_ALPHA.index(nt2)] = mult

    for pair1, pair2 in coupled_pairs:
        nt11, nt12 = pair1
        nt21, nt22 = pair2

        key1 = f"STCK_{nt11}_{nt12}"
        val1 = vals[key1]

        key2 = f"STCK_{nt21}_{nt22}"
        val2 = vals[key2]

        if enforce_symmetry:
            assert(val1 == val2)

        calc_f1_eps = val1 * (1.0 - stck_fact_eps + (kt * 9.0 * stck_fact_eps))
        mult = calc_f1_eps / sa_f1_eps
        stack_mult[DNA_ALPHA.index(nt11), DNA_ALPHA.index(nt12)] = mult

        calc_f1_eps = val2 * (1.0 - stck_fact_eps + (kt * 9.0 * stck_fact_eps))
        mult = calc_f1_eps / sa_f1_eps
        stack_mult[DNA_ALPHA.index(nt21), DNA_ALPHA.index(nt22)] = mult

    return hb_mult, stack_mult


def write_ss_oxdna(out_fpath, hb_mult, stack_mult,
                   f1_eps_hb=1.077,
                   f1_eps_base_stack=1.3448,
                   f1_eps_kt_coeff_stack=2.6568,
                   enforce_symmetry=True, t_kelvin=DEFAULT_TEMP,
                   round_places=6,
                   coupled_pairs=STCK_COUPLED_PAIRS_OXDNA1,
                   uncoupled_pairs=STCK_UNCOUPLED_PAIRS_OXDNA1):

    # Hydrogen bonding
    hb_lines = list()

    if enforce_symmetry:
        assert(hb_mult[DNA_ALPHA.index("A")][DNA_ALPHA.index("T")] == hb_mult[DNA_ALPHA.index("T")][DNA_ALPHA.index("A")])
        assert(hb_mult[DNA_ALPHA.index("G")][DNA_ALPHA.index("C")] == hb_mult[DNA_ALPHA.index("C")][DNA_ALPHA.index("G")])

    for nt1, nt2 in ["AT", "TA", "GC", "CG"]:
        mult = hb_mult[DNA_ALPHA.index(nt1)][DNA_ALPHA.index(nt2)]
        val = mult * f1_eps_hb
        hb_lines.append(f"HYDR_{nt1}_{nt2} = {onp.round(val, round_places)}")

    # Stacking
    stack_lines = list()
    kt = get_kt(t_kelvin) # Note: could be any value for kT within reason
    stck_fact_eps = f1_eps_kt_coeff_stack / (9*f1_eps_base_stack + f1_eps_kt_coeff_stack)

    sa_f1_eps = f1_eps_base_stack + kt * f1_eps_kt_coeff_stack


    for nt1, nt2 in uncoupled_pairs:
        ss_f1_eps = stack_mult[DNA_ALPHA.index(nt1)][DNA_ALPHA.index(nt2)] * sa_f1_eps
        val = ss_f1_eps / (1.0 - stck_fact_eps + (kt * 9.0 * stck_fact_eps))
        stack_lines.append(f"STCK_{nt1}_{nt2} = {onp.round(val, round_places)}")

    for pair1, pair2 in coupled_pairs:
        nt11, nt12 = pair1
        nt21, nt22 = pair2

        ss_f1_eps_p1 = stack_mult[DNA_ALPHA.index(nt11)][DNA_ALPHA.index(nt12)] * sa_f1_eps
        val_p1 = ss_f1_eps_p1 / (1.0 - stck_fact_eps + (kt * 9.0 * stck_fact_eps))

        ss_f1_eps_p2 = stack_mult[DNA_ALPHA.index(nt21)][DNA_ALPHA.index(nt22)] * sa_f1_eps
        val_p2 = ss_f1_eps_p2 / (1.0 - stck_fact_eps + (kt * 9.0 * stck_fact_eps))

        if enforce_symmetry:
            assert(val_p1 == val_p2)

        stack_lines.append(f"STCK_{nt11}_{nt12} = {onp.round(val_p1, round_places)}")
        stack_lines.append(f"STCK_{nt21}_{nt22} = {onp.round(val_p2, round_places)}")


    all_lines = [f"STCK_FACT_EPS = {onp.round(stck_fact_eps, round_places)}f"] + stack_lines + hb_lines

    with open(out_fpath, 'w') as f:
        f.write('\n'.join(all_lines) + "\n")


if __name__ == "__main__":
    fpath = "data/seq-specific/seq_oxdna1.txt"
    # fpath = "data/seq-specific/seq_oxdna2.txt"
    hb_mult, stack_mult = read_ss_oxdna(fpath)

    # write_ss_oxdna("test.txt", hb_mult, stack_mult)
    write_ss_oxdna("test.txt", hb_mult, stack_mult, t_kelvin=350)

    hb_mult = onp.random.rand(4, 4)
    stack_mult = onp.random.rand(4, 4)
    hb_mult, stack_mult = constrain(jnp.array(hb_mult), jnp.array(stack_mult))



    fpath = "data/seq-specific/seq_oxdna2.txt"
    hb_mult, stack_mult = read_ss_oxdna(fpath, coupled_pairs=STCK_COUPLED_PAIRS_OXDNA2)
