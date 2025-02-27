#!/bin/bash

# Linux requirements

apt install sudo

# >>>> HD-BET brain extraction tool

echo "Installing HD-BET brain extraction tool"

ss_folder="./preprocessing/SkullStripping_tool"

# Create the folder if it doesn't exist
mkdir -p "$ss_folder"
cd "$ss_folder" || { echo "Failed to change directory"; exit 1; }

if [ -d "./HD-BET" ]; then
    # Clone HD-BET repository
    echo "removing source"
    rm -rf ./HD-BET
fi

git clone https://github.com/MIC-DKFZ/HD-BET || { echo "Git clone failed"; exit 1; }


# Change into HD-BET directory
cd HD-BET || { echo "Failed to change directory to HD-BET"; exit 1; }

# Install using pip
pip3 install -e .

cd ..  # fix the from torch.cuda.amp import GradScaler
echo "current path ${pwd} "
bash nnUNET_fix_import.sh