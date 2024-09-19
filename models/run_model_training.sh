#!/bin/bash

# Navigate to the directory containing the Python script
conda init 
conda activate fair-chem
cd /media/mohammed/Work/FORMED_ML

output_dir=/media/mohammed/Work/FORMED_ML/models/training/logs
mkdir -p $output_dir
error_dir=/media/mohammed/Work/FORMED_ML/models/training/error
mkdir -p $error_dir
data_to_fit=log_S1_osc
# Run the Python script and redirect output to the directories
#python models/run_model_training_GNN.py --data_to_fit S1_exc --test --config_path config_files/equiformer_v2/equiformer_v2.yml > $output_dir/log_equiformer_v2.txt 2>$error_dir/error_equiformer_v2.txt
python models/run_model_training_GNN.py --data_to_fit $data_to_fit --config_path config_files/painn/painn.yml > $output_dir/log_painn.txt 2>$error_dir/error_painn.txt
python models/run_model_training_GNN.py --data_to_fit $data_to_fit --config_path config_files/schnet/schnet.yml > $output_dir/log_schnet.txt 2>$error_dir/error_schnet.txt
python models/train_xgboost_slatm.py --target $data_to_fit >$output_dir/output_xgb.txt 2>$error_dir/error_xgb.txt
python models/run_model_training_GNN.py --data_to_fit $data_to_fit --test --config_path config_files/equiformer_v2/equiformer_v2_N@12_L@6_M@2.yml > $output_dir/log_schnettxt 2>$output/error_schnet.txt
python models/run_model_training_GNN.py --data_to_fit $data_to_fit --test --config_path config_files/scn/scn.yml > $output_dir/log_scn.txt 2>$error_dir/error_scn.txt

