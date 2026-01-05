ARG VARIANT="3.10"
FROM mcr.microsoft.com/vscode/devcontainers/python:${VARIANT}
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglx-mesa0 \
    && rm -rf /var/lib/apt/lists/*
