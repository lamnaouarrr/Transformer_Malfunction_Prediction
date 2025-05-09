#!/usr/bin/env bash

# Project setup script for Transformer Malfunction Prediction
# Usage: source setup.sh

# Activate or create virtual environment
if [ -d transformer_venv ]; then
  source transformer_venv/bin/activate
else
  python3 -m venv transformer_venv
  source transformer_venv/bin/activate
fi

# Install project requirements
pip install --upgrade pip
pip install -r requirements.txt

# Install API requirements
pip install -r app/app_requirements.txt

# Export PYTHONPATH to include project root
export PYTHONPATH=$(pwd)

echo "Environment setup complete. Virtual environment activated."
