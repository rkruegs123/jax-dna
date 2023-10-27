#!/bin/bash

# Usage: bash ./scripts/mps.sh <path/to/oxdna/basedir> <iteration/basedir> <num. repeats>

# Read Arguments
oxdna_basedir=$1;
oxdna_exec_path="${oxdna_basedir}/build/bin/oxDNA";
iter_basedir=$2;
num_repeats=$3;

# export CUDA_MPS_PIPE_DIRECOTRY=$iter_basedir;
export CUDA_MPS_LOG_DIRECTORY=$iter_basedir;


sleep 2.0

# Setup MPS control
nvidia-cuda-mps-control -d

sleep 2.0

# Start the MPS jobs
for i in $(seq 0 $(($num_repeats-1))); do
    repeat_dir="${iter_basedir}/r${i}"
    input_path="${repeat_dir}/input"

    $oxdna_exec_path $input_path &
    sleep 0.25
done
wait
echo quit | nvidia-cuda-mps-control



