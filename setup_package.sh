#!/bin/bash

# Create a virtual environment
python3 -m venv houglass_destroyer

# Activate the virtual environment
source houglass_destroyer/bin/activate

# Install required packages
pip install -r requirements.txt

# Deactivate the virtual environment
deactivate

echo "Package installation completed. To activate the virtual environment, run 'source 
houglass_destroyer/bin/activate'."

