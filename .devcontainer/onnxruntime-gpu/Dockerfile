ARG VARIANT="3.10"
FROM nvcr.io/nvidia/pytorch:24.02-py3
ARG USERNAME=vscode
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglx-mesa0 \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m vscode \
    && apt-get update \
    && apt-get install -y sudo \
    && echo "vscode ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

USER ${USERNAME}
