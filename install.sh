#!/bin/bash

# Upgrade pip and install required Python packages
pip install --upgrade pip
pip install pandas datasets openai openpyxl transformers==4.48.3 bitsandbytes==0.45.0

# Clone the required repository
git clone -b v0.4.3.post2 https://github.com/sgl-project/sglang.git

# Navigate into the cloned directory
cd sglang || exit 1

# Install sgl-kernel with specific flags
pip install sgl-kernel --force-reinstall --no-deps

# Install sglang with dependencies
pip install -e "python[all]" --find-links https://flashinfer.ai/whl/cu124/torch2.5/flashinfer-python
