#!/bin/bash

# Navigate to the directory containing the Python script
cd /media/mohammed/Work/FORMED_ML
conda init
conda activate fair-chem
# Create output directories if they don't exist
mkdir -p output/logs
mkdir -p output/results
mkdir -p output/logs_equiformer
mkdir -p output/results_equiformer

# Run the Python script and redirect output to the directories
python3 optuna_equiformer.py > output/logs_equiformer/output.log 2> output/logs_equiformer/error.log
python3 test_optun_schnet.py > output/logs/output.log 2> output/logs/error.log
