#!/bin/bash
export SIF=isaac_sim_4.5.0.sif
export WORKDIR=$(pwd)/isaac_workspace

apptainer exec --nv \
    --bind "$WORKDIR:/isaac_workspace" \
    $SIF \
    /isaac-sim/python.sh /isaac_workspace/test.py


