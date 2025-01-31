#!/bin/bash

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

else

    # Active the venv
    source GPU/bin/activate
    
fi

source $HOME/.bashrc
