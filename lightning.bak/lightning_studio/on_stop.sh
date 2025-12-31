#!/bin/bash

# This script runs every time your Studio sleeps, from your home directory.

# Logs from previous runs can be found in ~/.lightning_studio/logs/

# Add your shutdown commands below.
#
# Example: docker down my-container
# Example: sudo service mysql stop

echo "[on_stop] cleaning..."
rm -rf /teamspace/studios/this_studio/.cache
rm -rf /teamspace/studios/this_studio/tmp_checkpoints
echo "[on_stop] done."
