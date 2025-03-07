#!/bin/bash

# Exit immediately if any command fails
set -e

# Update and upgrade system packages
echo "Updating system packages..."
sudo apt update && sudo apt upgrade -y

# Install Python3 and pip
echo "Installing Python and pip..."
sudo apt install -y python3 python3-pip python3-venv

# Install Git
echo "Installing Git..."
sudo apt install -y git

# create a folder
mkdir code
cd code

# Clone the repository (Change URL to your repo)
REPO_URL="https://github.com/Shireeshadeepthi/next-level.git"
echo "Cloning repository from $REPO_URL..."
git clone "$REPO_URL"
cd next-level/src  # Change to your repo's directory

# Install dependencies
echo "Installing dependencies from requirements.txt..."
pip install -r requirements.txt --index-url https://download.pytorch.org/whl/cu118


echo "Setup completed successfully!"

python3 content.py
