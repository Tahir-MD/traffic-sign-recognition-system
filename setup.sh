#!/bin/bash

# Install system dependencies for OpenCV
apt-get update
apt-get install -y libgl1-mesa-glx libglib2.0-0

# Install Python packages
pip install --upgrade pip
pip install -r requirements.txt