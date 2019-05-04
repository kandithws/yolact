FROM nvidia/cuda:9.0-cudnn7-devel

# Install some basic utilities
RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    sudo \
    git \
    bzip2 \
    # libx11-6 \
    libgtk2.0-dev \
    python3 \
    python3-pip \
    python3-tk \
 && rm -rf /var/lib/apt/lists/*

# Create a non-root user and switch to it
RUN adduser --disabled-password --gecos '' --shell /bin/bash user
RUN echo "user ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/90-user

RUN pip3 install torch torchvision
RUN pip3 install cython
RUN pip3 install opencv-python pillow pycocotools matplotlib
RUN pip3 install --upgrade pip 
RUN pip3 install grpcio grpcio-tools

# All users can use /home/user as their home directory
ENV HOME=/home/user
RUN chmod 777 /home/user
USER user
WORKDIR /home/user

CMD ["/bin/bash"]
