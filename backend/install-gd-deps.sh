#!/bin/bash

# Script to install GoDaddy-specific dependencies from Artifactory
# This script should be run with proper Artifactory credentials

set -e

echo "Installing GoDaddy-specific dependencies..."

# Check if Artifactory credentials are provided
if [ -z "$ARTIFACTORY_USERNAME" ] || [ -z "$ARTIFACTORY_PASSWORD" ]; then
    echo "Error: ARTIFACTORY_USERNAME and ARTIFACTORY_PASSWORD environment variables must be set"
    echo "Example:"
    echo "export ARTIFACTORY_USERNAME=your_username"
    echo "export ARTIFACTORY_PASSWORD=your_password"
    echo "Then run: ./install-gd-deps.sh"
    exit 1
fi

# Configure pip to use GoDaddy Artifactory
echo "Configuring pip to use GoDaddy Artifactory..."
python -m pip config set global.index-url https://${ARTIFACTORY_USERNAME}:${ARTIFACTORY_PASSWORD}@gdartifactory1.jfrog.io/artifactory/api/pypi/python-virt/simple
python -m pip config set global.trusted-host gdartifactory1.jfrog.io

# Install GoDaddy-specific dependencies
echo "Installing GoDaddy dependencies..."
python -m pip install -r additional-requirements.txt --no-cache-dir

# Clear the credentials from pip config after installation
echo "Cleaning up pip configuration..."
python -m pip config unset global.index-url
python -m pip config unset global.trusted-host

echo "GoDaddy dependencies installed successfully!"
echo "You can now run the Open WebUI application." 