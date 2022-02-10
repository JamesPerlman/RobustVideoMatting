# syntax=docker/dockerfile:1

FROM continuumio/miniconda3:4.10.3p0-alpine AS miniconda

FROM nvidia/cuda:11.4.1-cudnn8-devel-ubuntu20.04 AS main

# Set up CUDA build machine
ENV DEBIAN_FRONTEND noninteractive
ENV HOME_PATH=/usr/local
ENV PROJ_PATH=${HOME_PATH}/RobustVideoMatting

# We just need conda from the alpine miniconda build

COPY --from=miniconda /opt/conda /opt/conda

RUN echo "export PATH=\${PATH}:/opt/conda/bin" >> ~/.bashrc
ENV PATH=${PATH}:/opt/conda/bin

# Install apt dependencies first
RUN apt-get update && \
    apt-get install -y \
        git \
        wget

# Clone repo

RUN git clone https://github.com/JamesPerlman/RobustVideoMatting.git ${PROJ_PATH}

WORKDIR ${PROJ_PATH}

# Fetch model

RUN mkdir models && \
    cd models && \
    wget https://github.com/PeterL1n/RobustVideoMatting/releases/download/v1.0.0/rvm_resnet50.pth

# Create conda env

RUN conda create -n RobustVideoMatting python=3.8 && \
    echo "source activate RobustVideoMatting" >> ~/.bashrc

SHELL ["conda", "run", "-n", "RobustVideoMatting", "/bin/bash", "-c"]

RUN pip install -r requirements_inference.txt

