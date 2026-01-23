#!/bin/bash

# --- Configuration ---
LOCAL_DIR="/home/guglielmo/Uni/hpc-project/"
REMOTE_USER="guglielmo.boi"
REMOTE_HOST="hpc3-login1"
REMOTE_DIR="/home/guglielmo.boi/hpc-project/"

# --- Rsync command ---
rsync -avz --delete \
  --exclude=".git/" \
  --exclude=".venv/" \
  --exclude=".vscode/" \
  --exclude="bin/" \
  --exclude="build/" \
  "$LOCAL_DIR" \
  "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}"
