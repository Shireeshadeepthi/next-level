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

# Clone the repository (Change URL to your repo)
REPO_URL="https://github.com/yourusername/your-repo.git"
echo "Cloning repository from $REPO_URL..."
git clone "$REPO_URL"
cd your-repo  # Change to your repo's directory

# Install dependencies
echo "Installing dependencies from requirements.txt..."
pip3 install -r requirements.txt

echo "Setup completed successfully!"
