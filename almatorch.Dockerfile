FROM almalinux:9
RUN dnf update -y && \
    dnf install -y epel-release && \
    dnf install -y \
        python3 \
        python3-pip \
        libjpeg-dev \
        git \
        gcc \
        gcc-c++ \
        wget \
        which && \
    dnf clean all

RUN pip3 install --upgrade pip
RUN pip3 install wheel setuptools
RUN pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/rocm7.0

CMD ["/bin/bash"]