# Apptainer_RL
A simple container for using MuJoCo-mjx, and Jax on the GPU of your PC. Installs Cuda and CudNN as well as other python dependancies.

## How to use
The only requirment for using this repository is [Singularity/Apptainer](https://apptainer.org/admin-docs/master/installation.html) Version >= 3.5.

Once Singularity/Apptainer has been installed, you can simply build the container using /.build.sh (This will take a while depending on your machine and wifi connection, approximately 30 minutes) and then you can launch the container using /.run.sh.

If you need to admin priveledges inside the container to isntall something, you can use /.write.sh.

## Current Issues
- Repository is hard coded to download Cuda 12.4 and Cudnn 9.5. I will make this flexible once I figure that out.
- Some possible issue with jax installation. It seems to be using the GPU but there is a warning about a version mismatch between cuda and XLA.

# ToDo
- Add paths in post script
- Make CUDA and CuDDn installations dynamic.
- Look into fixing XLA and Jax mismatch?
- Make some nice mjx-examples and have this repo clone them automatically.
