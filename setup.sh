#!/bin/bash
set -e

# Set environment
export PYTHONUNBUFFERED=1
export PIP_DISABLE_PIP_VERSION_CHECK=1
export PIP_NO_CACHE_DIR=1

# Print versions for debugging
echo "=== System Information ==="
uname -a
python --version
pip --version

# Function to install with retries
install_with_retry() {
    local max_attempts=3
    local delay=5
    local attempt=1
    local exit_code=0

    while [ $attempt -le $max_attempts ]; do
        echo "Attempt $attempt of $max_attempts: $@"
        if "$@"; then
            echo "Installation successful"
            return 0
        else
            exit_code=$?
            echo "Installation failed with exit code $exit_code, retrying in $delay seconds..."
            sleep $delay
            ((attempt++))
        fi
    done

    echo "Failed after $max_attempts attempts"
    return $exit_code
}

# Install system dependencies if on Debian/Ubuntu
if command -v apt-get >/dev/null 2>&1; then
    echo "Installing system dependencies..."
    apt-get update && apt-get install -y \
        python3-dev \
        python3-distutils \
        build-essential \
        libopenblas-dev
fi

# Upgrade pip and setuptools
echo "Upgrading pip and setuptools..."
install_with_retry python -m pip install --upgrade pip setuptools wheel

# Install numpy first as it's a common build dependency
echo "Installing numpy..."
install_with_retry pip install "numpy>=1.24.0,<2.0.0" --progress-bar off

# Install PyTorch with CPU-only version
echo "Installing PyTorch..."
install_with_retry pip install --no-cache-dir \
    torch==2.1.2+cpu \
    torchvision==0.16.2+cpu \
    torchaudio==2.1.2+cpu \
    --index-url https://download.pytorch.org/whl/cpu \
    --progress-bar off

# Install other requirements first
if [ -f "requirements.txt" ]; then
    echo "Installing Python requirements..."
    install_with_retry pip install -r requirements.txt --progress-bar off
fi

# Install spaCy model using spacy download
echo "Installing spaCy model using spacy download..."
python -m spacy download en_core_web_sm --no-cache-dir

# Verify the model can be loaded
if ! python -c "import spacy; spacy.load('en_core_web_sm')" &>/dev/null; then
    echo "Warning: Could not load spaCy model, trying direct download..."
    # Fallback: Download and install the model directly
    wget -q https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.6.0/en_core_web_sm-3.6.0.tar.gz
    pip install --no-cache-dir en_core_web_sm-3.6.0.tar.gz
    rm en_core_web_sm-3.6.0.tar.gz
    
    # Verify again after direct install
    if ! python -c "import spacy; spacy.load('en_core_web_sm')" &>/dev/null; then
        echo "Warning: Could not load spaCy model, but continuing..."
        echo "The application will attempt to use a fallback model at runtime."
    fi
fi

# Download NLTK data
echo "Downloading NLTK data..."
install_with_retry python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"

# Verify installations
echo -e "\n=== Installed Packages ==="
pip list

echo -e "\n✅ Setup completed successfully!"
