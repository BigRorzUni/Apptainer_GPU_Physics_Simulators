#!/bin/bash

if [ ! -d "$HOME/newton" ]; then
    # MPC
    git clone git@github.com:BigRorzUni/NewtonMPC.git

#    git clone https://github.com/DMackRus/mujoco_models.git

    # Create a virtual environment
    python3 -m venv GPU

    # Active the venv
    source GPU/bin/activate

    pip install --upgrade pip

    # Install all python packages into the venv
    pip install matplotlib
    pip install torch torchvision torchaudio 
    pip install pandas
#    pip install genesis-world


    # Install mujoco and mujoco viewer
    pip install mujoco
    pip install mujoco-python-viewer

    pip install git+https://github.com/Genesis-Embodied-AI/Genesis.git
    # Install jax and jaxlib
    pip install -U "jax[cuda12]"
    #pip install --upgrade pip
    #pip install --upgrade "jax[cuda12]"

    # Install mujoco-mjx
    pip install mujoco-mjx
    pip install brax

    pip install pytest

    #  Warp
    pip install uv
    uv pip install newton-physics

    git clone https://github.com/newton-physics/newton
    cd newton
    python -m pip install mujoco --pre -f https://py.mujoco.org/
    python -m pip install warp-lang --pre -U -f https://pypi.nvidia.com/warp-lang/
    python -m pip install git+https://github.com/google-deepmind/mujoco_warp.git@main
    python -m pip install -e .[dev]
    cd ..
else

    # Active the venv
    source GPU/bin/activate

fi

source $HOME/.bashrc
