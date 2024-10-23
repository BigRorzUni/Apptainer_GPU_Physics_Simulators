#!/bin/bash

apt-get update 
DEBIAN_FRONTEND=noninteractive apt-get install -y keyboard-configuration
DEBIAN_FRONTEND=noninteractive TZ="Europe/London" apt-get install -y tzdata
apt-get -y upgrade

apt-get install -y \
    wget \
    unzip \
    git \
    build-essential \
    curl \
    python-is-python3 \
    python3-pip \
    mesa-utils \
    gpg
  
  
#Install VScode
apt-get update
wget -qO- https://packages.microsoft.com/keys/microsoft.asc | gpg --dearmor | tee /usr/share/keyrings/microsoft.gpg > /dev/null
sh -c 'echo "deb [arch=amd64 signed-by=/usr/share/keyrings/microsoft.gpg] https://packages.microsoft.com/repos/code stable main" > /etc/apt/sources.list.d/vscode.list'
apt-get update && apt-get install -y code

# --------------- Install CuDa / nvcc ----------------
# Base installer
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/12.4.0/local_installers/cuda-repo-ubuntu2204-12-4-local_12.4.0-550.54.14-1_amd64.deb
dpkg -i cuda-repo-ubuntu2204-12-4-local_12.4.0-550.54.14-1_amd64.deb
cp /var/cuda-repo-ubuntu2204-12-4-local/cuda-*-keyring.gpg /usr/share/keyrings/
apt-get update
apt-get -y install cuda-toolkit-12-4

# Driver installer (possibly untested)
apt-get install -y cuda-drivers

# -----------------    Install Cudnn    --------------------
# Some prerequisite
apt-get install zlib1g
# Main installation
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
dpkg -i cuda-keyring_1.1-1_all.deb
apt-get update
apt-get -y install cudnn9-cuda-12

# ---------------- Install pip packages --------------------
# Install mujoco and mujoco viewer
pip install mujoco
pip install mujoco-python-viewer

# Install jax and jaxlib
pip install --upgrade pip
pip install --upgrade "jax[cuda12]"

# Install mujoco-mjx
pip install mujoco-mjx
pip install brax

# Install Matplotlib
pip install matplotlib


# Let's have a custom PS1 to help people realise in which container they are
# working.
CUSTOM_ENV=/.singularity.d/env/99-zz_custom_env.sh
cat >$CUSTOM_ENV <<EOF
#!/bin/bash
PS1="[Apptainer_MuJoCo-mjx] Singularity> \w \$ "
EOF
chmod 755 $CUSTOM_ENV

