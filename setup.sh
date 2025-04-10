#!/bin/bash

# Create nltk_data directory
mkdir -p nltk_data

# Download required corpora
python -m nltk.downloader -d nltk_data punkt stopwords

