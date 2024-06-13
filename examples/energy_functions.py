
import jax_md
import jax_dna.energy.dna1 as dna1

displacement_fn ,_ = jax_md.space.free()

args = (
    displacement_fn,
)


fn1 = dna1.Fene_DG(displacement_fn)
fn2 = dna1.Bonded_DG(displacement_fn)
fn3 = dna1.Unbonded_DG(displacement_fn)
fn4 = dna1.HB_DG(displacement_fn)
fn5 = dna1.Stacking_DG(displacement_fn)
fn6 = dna1.CrossStacking_DG(displacement_fn)
fn7 = dna1.CoaxialStacking_DG(displacement_fn)

default_dna1 = fn1 + fn2 + fn3 + fn4 + fn5 + fn6 + fn7


print("A Single Energy Function =============")
print(fn1)
print("======================================")
print("A composed energy function ===========")
print(fn1 + fn2)
print("======================================")
print("A composed energy function ===========")
print(fn3 * 3)
print("======================================")
print("A composed energy function ===========")
print(fn4 + fn5 + fn6 + fn7)
print("======================================")
print("default oxdna ========================")
print(sum([fn2, fn3, fn4, fn5, fn6, fn7], start=fn1))
print("or")
print(fn1 + fn2 + fn3 + fn4 + fn5 + fn6 + fn7)
print("======================================")

