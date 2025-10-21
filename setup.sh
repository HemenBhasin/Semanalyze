#!/bin/bash
set -e

# Upgrade pip with progress bar
python -m pip install --upgrade pip --progress-bar off

# Install requirements with progress bar
pip install -r requirements.txt --progress-bar off

# Install PyTorch CPU-only (lighter weight)
pip install torch==1.13.1+cpu torchvision==0.14.1+cpu torchaudio==0.13.1 \
    --index-url https://download.pytorch.org/whl/cpu --progress-bar off

# Install sentence-transformers and spacy
pip install sentence-transformers==2.2.2 spacy==3.6.1 --progress-bar off

# Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"

# Download spaCy model
python -m spacy download en_core_web_sm

echo "Setup completed successfully!"
