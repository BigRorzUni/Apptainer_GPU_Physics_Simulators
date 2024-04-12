-c /bin/bash

apt-get update 
DEBIAN_FRONTEND=noninteractive apt-get install -y keyboard-configuration
DEBIAN_FRONTEND=noninteractive TZ="Europe/London" apt-get install -y tzdata
apt-get -y upgrade

apt-get install -y --force-yes \
    wget \
    unzip \
    git \
    build-essential \
    curl \
    python \
    python3-pip \
    mesa-utils \
  
#Install VScode
cd /tmp
curl -o code.deb -L http://go.microsoft.com/fwlink/?LinkID=760868
apt-get install -y ./code.deb

# Install firefox
#apt install -y --force-yes firefox

# Let's have a custom PS1 to help people realise in which container they are
# working.
CUSTOM_ENV=/.singularity.d/env/99-zz_custom_env.sh
cat >$CUSTOM_ENV <<EOF
#!/bin/bash
PS1="[Apptainer_RL] Singularity> \w \$ "
EOF
chmod 755 $CUSTOM_ENV

