# syntax = docker/dockerfile:experimental

FROM tensorflow/tensorflow:2.3.1-gpu

# Install a few necessary need or useful libs and apps
RUN apt update && apt install -y wget vim libgl1-mesa-dev ack

# Download base tensorflow model to app_user's folder, change permission so he can use it
RUN mkdir -p /root/.keras/models && \
    wget https://storage.googleapis.com/tensorflow/keras-applications/resnet/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5 \
        -O /root/.keras/models/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5

# Install python environment
COPY ./requirements.txt /tmp/requirements.txt
RUN --mount=type=cache,mode=0777,target=/root/.cache/pip pip install -r /tmp/requirements.txt

# Setup bashrc
COPY ./docker/bashrc /root/.bashrc

# Setup PYTHONPATH
ENV PYTHONPATH=.

# Tensorflow keeps on using deprecated APIs ^^
ENV PYTHONWARNINGS="ignore::DeprecationWarning:tensorflow"

# # Set up working directory
WORKDIR /app
