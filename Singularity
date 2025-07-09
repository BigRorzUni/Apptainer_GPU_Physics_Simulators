bootstrap:docker
From:ubuntu:24.04

%labels

AUTHOR David Russell

%environment
    export LANG=C.UTF-8
    export LC_ALL=C.UTF-8
    export XDG_RUNTIME_DIR=/tmp  # This fixes an issue with /run read-only permissions when using Rviz and other tools.
    export XLA_PYTHON_CLIENT_PREALLOCATE=false # Fixes issue with jax when running alongside other gpu applications
    export PYTHONPATH=~/test/helpers:$PYTHONPATH

%files
  scripts /scripts

%post
  /scripts/post_script.sh

%runscript
  exec /bin/bash "$@" --rcfile /scripts/run_script.sh
