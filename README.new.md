# JAX-DNA
[![CI](https://github.com/rkruegs123/jax-dna/actions/workflows/ci.yml/badge.svg)](https://github.com/rkruegs123/jax-dna/actions/workflows/ci.yml)
[![Documentation Status](https://readthedocs.org/projects/jax-dna/badge/?version=latest)](https://jax-dna.readthedocs.io/en/latest/)
[![codecov](https://codecov.io/gh/rkruegs123/jax-dna/branch/master/graph/badge.svg?token=0KPNKHRC2V)](https://codecov.io/gh/rkeugs123/jax-dna)
[![Security](https://github.com/rkreugs123/jax-dna/actions/workflows/security.yml/badge.svg)](https://github.com/rkruegs123/jax-dna/actions/workflows/security.yml)
[![arXiv](https://img.shields.io/badge/arXiv-2411.09216-b31b1b.svg)](https://arxiv.org/abs/2411.09216)


> [!NOTE]
> This project is in active development, so expect the API to change frequently.

JAX-DNA is a Python package for simulating and fitting coarse-grained molecular
models to macroscopic experimental data.

Currently, JAX-DNA can run simulations using
[JAX-MD](https://github.com/jax-md/jax-md) and [oxDNA](https://oxdna.org/)
(You must install oxDNA yourself however).

Further, JAX-DNA supports the fitting models using JAX-MD (Direct Differentiation,
and [DiffTRe](https://www.nature.com/articles/s41467-021-27241-4)) and oxDNA
(DiffTRe only).


## Quick Start

We recommend using a fresh conda environment with Python 3.11. You can create a
new environment with the following command:

```bash
conda create -y -n jax-dna python=3.11
conda activate jax-dna
```

Depending on your hardware, you may want to install the GPU accelerated version
of JAX, see the [JAX
documentation](https://docs.jax.dev/en/latest/installation.html#installation)
for more details on how to do this. If you aren't interested in GPU support, you
can skip straight to installing JAX-DNA which will install the CPU version of
JAX.


First install JAX-DNA using pip:

```bash
pip install git+https://github.cpom/rkruegs123/jax-dna.git
```

### Simulations

Information on how to run a simulation can be found in the
[documentation](https://jax-dna.readthedocs.io/en/latest/basic_usage.html#running-a-single-simulation).

One advantage of JAX-DNA is that you can specify a custom energy function for
both simulations and optimizations. Information on how energy functions are
defined and how to define your own energy functions can be found in the
documentation.

### Optimizations

Optimizations can be thought of in two kinds: simple and advanced.


#### Simple Optimizations

Simple optimizations are those optimizations where a set of parameters are fit
with respect to a single objective as a function of a single simulation. This is
usually a helpful introduction to how JAX-DNA works. Documentation on how simple
optimizations work can be found
[here](https://jax-dna.readthedocs.io/en/latest/basic_usage.html#running-a-simple-optimization)
in the documentation.

To see examples of simple simulations, go to
[examples](https://github.com/rkruegs123/jax-dna/tree/master/examples/simulations),
where you can find both JAX-MD and oxDNA examples.

### Advanced Optimizations

Advanced optimizations are those optimizations where a set of parameters are
fit with respect to more than one objective and as a function of multiple possibly
heterogeneous simulations. These kinds of optimizations are covered in the
documentation [here](https://jax-dna.readthedocs.io/en/latest/advanced_usage.html#advanced-optimizations).


To see examples of simple simulations go to see
[examples](https://github.com/rkruegs123/jax-dna/tree/master/examples/simulations),
where you can find both JAX-MD and oxDNA examples.


## Development

We welcome contributions! If you are looking for something to work on, check out
the [issues](https://github.com/rkruegs123/jax-dna/issues).

If you have a feature request or an idea that you would like to contribute,
please open an issue. The project is fast moving, opening an issue will help us
to give you quick feedback and help you to get started.

See the [CONTRIBUTING](https://github.com/rkruegs123/jax-dna/blob/master/CONTRIBUTING.md)


