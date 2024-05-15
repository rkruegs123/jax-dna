# ruff: noqa
# fmt: off
import pdb
from pathlib import Path
import numpy as onp

from jax_dna.common.utils import DNA_ALPHA, get_kt, DEFAULT_TEMP


def read_ss_oxdna(
        ss_path,
        default_f1_eps_hb=1.077,
        default_f1_eps_base_stack=1.3448,
        default_f1_eps_kt_coeff_stack=2.6568,
        enforce_symmetry=True,
        t_kelvin=DEFAULT_TEMP
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

    uncoupled_vals = ["GC", "CG", "AT", "TA"]
    for nt1, nt2 in uncoupled_vals:
        key_name = f"STCK_{nt1}_{nt2}"
        val = vals[key_name]
        calc_f1_eps = val * (1.0 - stck_fact_eps + (kt * 9.0 * stck_fact_eps))
        mult = calc_f1_eps / sa_f1_eps

        stack_mult[DNA_ALPHA.index(nt1), DNA_ALPHA.index(nt2)] = mult

    coupled_vals = [("GG", "CC"), ("GA", "TC"), ("AG", "CT"),
                    ("TG", "CA"), ("GT", "AC"), ("AA", "TT")]
    for pair1, pair2 in coupled_vals:
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


if __name__ == "__main__":
    fpath = "data/seq-specific/seq_oxdna1.txt"
    hb_mult, stack_mult = read_ss_oxdna(fpath)

    pdb.set_trace()
