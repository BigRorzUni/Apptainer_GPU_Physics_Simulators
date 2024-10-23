#!/bin/bash

if [ ! -d "$HOME/MuJoCo_GPU_Learning" ]; then
    git clone git@github.com:DMackRus/MuJoCo_GPU_Learning.git
fi

source $HOME/.bashrc
