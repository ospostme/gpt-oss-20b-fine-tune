#!/bin/bash

# This script runs every time your Studio starts, from your home directory.

# Logs from previous runs can be found in ~/.lightning_studio/logs/

# List files under fast_load that need to load quickly on start (e.g. model checkpoints).
#
# ! fast_load
# <your file here>

# Add your startup commands below.
#
# Example: streamlit run my_app.py
# Example: gradio my_app.py
echo "[on_start] starting at $(date)"
ln -s /teamspace/studios/this_studio/.jupyter_kernels/share/jupyter/kernels/ml  /opt/jupyter/envs/main/share/jupyter/kernels/ml
echo "[on_start] done at $(date)"
