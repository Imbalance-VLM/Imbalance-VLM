#!/bin/bash

# Traverse all directories in /home/yzh/saved_models_backup/
for dir in /home/yzh/saved_models_backup/*/
do
    # Extract the directory name from the path
    dir_name=$(basename "$dir")
    
    # Construct the command using the directory name
    command="python eval.py --load_path /home/yzh/saved_models_backup/${dir_name}/model_best.pth"
    
    # Append the command to a shell script file
    echo "${command}" >> ../eval_all.sh
done
