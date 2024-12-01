ARG VARIANT="3.10"
FROM nvcr.io/nvidia/pytorch:24.02-py3
ARG USERNAME=vscode
RUN apt-get update && apt install -y libgl1-mesa-glx

RUN useradd -m vscode \
    && apt-get update \
    && apt-get install -y sudo \
    && echo "vscode ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

USER ${USERNAME}