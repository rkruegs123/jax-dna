import scipy
import pdb
import numpy as np

from jax_dna.cgdna import cgdna


def compute_marginals(seq, verbose=False):
    """
    Computes the marginal intra and inter coordinates for a sequence via cgDNA+

    The main code for the cgDNA+ model computes both base and phosphate parameters.
    The order is intra--phos--inter--phos--intra--phos--inter--phos-- ... --intra

    Note: each set of six coordinates is 3 rotations (in units of rad/5 ~ 11.5
    degrees) followed by 3 translations (in units of Angstroms), or vice versa
    """

    seq = seq.upper()
    n = len(seq)
    data = cgdna.cgDNA(seq, "dna_ps2")

    # Take the marginal over phosphates to obtain intra- and inter-coordinates
    cov = scipy.sparse.linalg.inv(data.stiff)  # dimension is (24N-18) x (24N-18) -- see thesis for details
    cov = cov.todense()
    cov_diag = np.diag(cov) # essentially the square of std. dev.
    mean = data.ground_state

    assert(len(mean) == 24 * n - 18) # note: 24N-18 = 6*(4N-3)
    assert(len(cov_diag) == 24 * n - 18)

    # mean and cov_diag are two vectors of dimension 24N-18 = 6*(4N-3 )
    # to marginalize, we keep the first 6 entries, then remove the next 6, then keep the next 6, and repeat, until we keep the last 6
    marginalized_mean = list()
    marginalized_cov_diag = list()
    num_6_chunks = len(mean) // 6
    for num_chunk in range(num_6_chunks):
        if num_chunk % 2 == 0:
            chunk_start_idx = num_chunk*6
            chunk_end_idx = chunk_start_idx + 6
            marginalized_mean += list(mean[chunk_start_idx:chunk_end_idx])
            marginalized_cov_diag += list(cov_diag[chunk_start_idx:chunk_end_idx])

    """
    Length of the marginalized vectors should be 6*(2n-1)
    - all internal and transition coordinates (6 each), no transition coordinates for first bp
    - order is intra--inter--intra--inter-- ... --intra

    With R=Rotation and T=Translation, order of coordinates is as follows:
    - intra (R, R, R, T, T, T): buckle, propeller, opening, shear, stretch, stagger
    - inter (T, T, T, R, R, R): tilt, roll, twist shift, slide, rise -- FIXME: check this... T vs. R is what Rahul said, but conflits with the tilt->rise ordering in his thesis
    """
    assert(len(marginalized_mean) == 6*(2*n-1))
    assert(len(marginalized_cov_diag) == 6*(2*n-1))

    # Print the coordinates for each base pair
    bp_map = {'G': 'C', 'C': 'G', 'A': 'T', 'T': 'A'}
    intra_coord_names = ["buckle", "propeller", "opening", "shear", "stretch", "stagger"]
    inter_coord_names = ["tilt", "roll", "twist", "shift", "slide", "rise"]
    intra_coord_means_map = {ic_name: list() for ic_name in intra_coord_names}
    intra_coord_vars_map = {ic_name: list() for ic_name in intra_coord_names}
    for bp_idx, base in enumerate(seq):
        pair = bp_map[base]
        if verbose:
            print(f"\nBase Pair {bp_idx}: {base+pair}")

        if bp_idx != 0:
            inter_coords_start_idx = (bp_idx*2 - 1)*6
            inter_coords_end_idx = inter_coords_start_idx + 6
            inter_coord_means = marginalized_mean[inter_coords_start_idx:inter_coords_end_idx]
            inter_coord_vars = marginalized_cov_diag[inter_coords_start_idx:inter_coords_end_idx]

            if verbose:
                print(f"- Inter coordinates")
            for tc_idx, (tc_val, tc_var) in enumerate(zip(inter_coord_means, inter_coord_vars)):
                tc_name = inter_coord_names[tc_idx]
                if verbose:
                    if tc_idx < 3:
                        # Rotation
                        # print(f"\t- {tc_name}: {np.round(tc_val * 11.5**2, 3)} degrees, (variance: {np.round(tc_var * 11.5, 3)})")
                        print(f"\t- {tc_name}: {np.round(tc_val, 3)} rad/5, (variance: {np.round(tc_var, 3)})")
                    else:
                        print(f"\t- {tc_name}: {np.round(tc_val, 3)} angstroms, (variance: {np.round(tc_var, 3)})")



        intra_coords_start_idx = (bp_idx*2)*6
        intra_coords_end_idx = intra_coords_start_idx + 6
        # print(f"- indices: {intra_coords_start_idx} to {intra_coords_end_idx-1} out of {len(marginalized_mean)}")
        intra_coord_means = marginalized_mean[intra_coords_start_idx:intra_coords_end_idx]
        intra_coord_vars = marginalized_cov_diag[intra_coords_start_idx:intra_coords_end_idx]

        # Print the intra coordinates
        if verbose:
            print(f"- Intra coordinates:")
        for ic_idx, (ic_val, ic_var) in enumerate(zip(intra_coord_means, intra_coord_vars)):
            ic_name = intra_coord_names[ic_idx]
            if verbose:
                if ic_idx < 3:
                    # Rotation
                    ic_std = np.sqrt(ic_var) # (rad/5)
                    ic_std_degrees = ic_std * 11.5
                    ic_var_degrees = ic_std_degrees ** 2
                    print(f"\t- {ic_name}: {np.round(ic_val * 11.5, 3)} degrees, (variance: {np.round(ic_var_degrees, 3)})")
                    # print(f"\t- {ic_name}: {np.round(ic_val, 3)} rad/5, (variance: {np.round(ic_var, 3)})")
                else:
                    print(f"\t- {ic_name}: {np.round(ic_val, 3)} angstroms, (variance: {np.round(ic_var, 3)})")

            intra_coord_means_map[ic_name].append(ic_val)
            intra_coord_vars_map[ic_name].append(ic_var)
    return intra_coord_means_map, intra_coord_vars_map
