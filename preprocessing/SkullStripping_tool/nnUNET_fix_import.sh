#!/bin/bash

# Path might need to be adjusted based on your Python environment
file_path="/usr/local/lib/python3.10/dist-packages/nnunetv2/training/nnUNetTrainer/nnUNetTrainer.py"

# Check if the file exists
if [ -f "$file_path" ]; then
  # Use sed to replace the import statement
  sudo sed -i 's/from torch import GradScaler/from torch.cuda.amp import GradScaler/' "$file_path"
  echo "Fixed GradScaler import in nnUNetTrainer.py line 40 in ${file_path}"
else
  echo "File not found: $file_path"
fi