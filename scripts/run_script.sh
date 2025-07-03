#!/bin/bash

ISAAC_SIM_SIF="$HOME/isaac_sim_4.5.0.sif"

if [ ! -d "$HOME/MuJoCo_GPU_Learning" ]; then
    git clone git@github.com:DMackRus/MuJoCo_GPU_Learning.git
    
    # Create a virtual environment
    python3 -m venv GPU
    
    # Active the venv
    source GPU/bin/activate
    
    # Install all python packages into the venv
    pip install matplotlib
    pip install torch torchvision torchaudio 
    pip install genesis-world
    
    # Install mujoco and mujoco viewer
    pip install mujoco
    pip install mujoco-python-viewer
    
    # Install jax and jaxlib
    pip install -U "jax[cuda12]"
    #pip install --upgrade pip
    #pip install --upgrade "jax[cuda12]"

    # Install mujoco-mjx
    pip install mujoco-mjx
    pip install brax
    
    pip install pytest

else

    # Active the venv
    source GPU/bin/activate
    
fi

source $HOME/.bashrc
echo "PATH after sourcing .bashrc: $PATH"
export PATH="/usr/local/bin:$PATH"
echo "PATH after prepending /usr/local/bin: $PATH"
ls -l /usr/local/bin
echo "Starting Isaac Sim"
/usr/local/bin/apptainer exec --nv "$ISAAC_SIM_SIF" ./runheadless.native.sh
