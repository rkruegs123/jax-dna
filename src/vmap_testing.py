import pdb

from jax import vmap
import jax.numpy as jnp

from jax_md import util


f32 = util.f32

if __name__ == "__main__":
    seq = jnp.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1]
    ], dtype=f32)

    nbs = [(0, 2), (1, 3), (0, 3), (2, 4)]
    nbs = jnp.array(nbs)

    nbs_i = nbs[:, 0]
    nbs_j = nbs[:, 1]

    pdb.set_trace()

    def foo(seq, i, j):
        return jnp.kron(seq[i], seq[j])

    get_hb_probs = vmap(foo, in_axes=(None, 0, 0), out_axes=0)

    hb_probs = get_hb_probs(seq, nbs_i, nbs_j) # get the probabilities of all possibile hydrogen bonds for all neighbors

    # Kron: AA, AC, AG, AT, CA, CC, CG, CT, GA, GC, GG, GT, TA, TC, TG, TT
    HB_WEIGHTS = jnp.array([
        0.0, 0.0, 0.0, 1.0, # AX
        0.0, 0.0, 1.0, 0.0, # CX
        0.0, 1.0, 0.0, 0.0, # GX
        1.0, 0.0, 0.0, 0.0  # TX
    ])

    hb_weights = jnp.dot(hb_probs, HB_WEIGHTS)

    # FIXME: OK, then just compute HB for all and multiply by the above weights!
