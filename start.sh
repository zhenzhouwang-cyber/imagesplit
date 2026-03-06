#!/bin/bash
cd "$(dirname "$0")"
echo "Installing dependencies..."
pip install -r requirements.txt
echo ""
echo "Starting Image Tool..."
python gui.py
