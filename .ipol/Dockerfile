# use one of the images from this repository: https://github.com/centreborelli/ipol-docker-images/
FROM registry.ipol.im/ipol:v1-py3.9

# copy the code to $bin
ENV bin /workdir/bin
RUN mkdir -p $bin
WORKDIR $bin
COPY . .

#RUN make

RUN wget https://mever.iti.gr/visloc/back_coll_features.hdf5

# the execution will happen in the folder /workdir/exec
# it will be created by IPOL

# some QoL tweaks
ENV PYTHONDONTWRITEBYTECODE 1
ENV PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION python
ENV PATH $bin:$PATH

# $HOME is writable by the user `ipol`, but 
ENV HOME /home/ipol
# chmod 777 so that any user can use the HOME, in case the docker is run with -u 1001:1001
RUN groupadd -g 1000 ipol && useradd -m -u 1000 -g 1000 ipol -d $HOME && chmod -R 777 $HOME
# Install system dependencies (for things like matplotlib and PyTorch)
RUN apt-get update && apt-get install -y \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
RUN pip install --upgrade pip

RUN pip install \
    torch torchvision \
    h5py \
    numpy \
    tqdm \
    scikit-learn \
    efficientnet_pytorch==0.7.0 \
    einops \
    s2sphere \
    matplotlib
USER ipol