#!/bin/bash
set -e

# Set environment variables
export PYTHONUNBUFFERED=1
export PIP_DISABLE_PIP_VERSION_CHECK=1

# Print Python and pip versions
python --version
pip --version

# Install system dependencies (will work on Debian-based systems)
if command -v apt-get >/dev/null 2>&1; then
    echo "Installing system dependencies..."
    apt-get update && apt-get install -y python3-distutils
fi

# Upgrade pip with retry mechanism
for i in {1..3}; do
    if python -m pip install --upgrade pip --progress-bar off; then
        break
    fi
    echo "Retrying pip upgrade..."
    sleep 5
done

# Install build dependencies
pip install --upgrade setuptools wheel --progress-bar off

# Install numpy first as it's a common build dependency
pip install "numpy>=1.26.0" --progress-bar off

# Install other requirements
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt --progress-bar off
fi

# Install PyTorch with CPU-only version
pip install --no-cache-dir \
    torch==2.1.0+cpu \
    torchvision==0.16.0+cpu \
    torchaudio==2.1.0 \
    --index-url https://download.pytorch.org/whl/cpu \
    --progress-bar off

# Install sentence-transformers and spacy
pip install --no-cache-dir \
    sentence-transformers==2.2.2 \
    spacy==3.6.1 \
    --progress-bar off

# Download NLTK data with retry
for i in {1..3}; do
    if python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"; then
        break
    fi
    echo "Retrying NLTK downloads..."
    sleep 10
done

# Download spaCy model with retry
for i in {1..3}; do
    if python -m spacy download en_core_web_sm; then
        break
    fi
    echo "Retrying spaCy model download..."
    sleep 10
done

# Verify installations
echo "\n=== Installed Packages ==="
pip freeze

echo -e "\n✅ Setup completed successfully!"
