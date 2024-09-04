#!/bin/bash

# This script runs the deepfly3d command line interface (df3d-cli) 
# on the images in the specified trial directory and saves the output
# The script is called by the run_behavior.sh script.
# If the script fails, it will return a non-zero exit code (1=256 = missing arguments, 2=512 = df3d-cli failed)

# Access the passed arguments:
# $1: path to the folder containing the folders to be processed
trial_dir=$1
# $2: path to the output folder (from user_config)
output_dir=$2
# $3: order of the cameras in the scope (from user_config)
camera_ids=$3

# Check if all required arguments are provided
if [[ -z "$trial_dir" || -z "$output_dir" || -z "$camera_ids" ]]; then
  echo "Error: Missing required arguments."
  echo "Usage: $0 <trial_dir> <output_dir> <df3d_env> <camera_ids> <overwrite>"
  exit 1
fi

# Make the conda command known to the current shell
BASE=$(conda info | grep -i 'base environment' | cut -d':' -f 2 | cut -d'(' -f 1 | cut -c2- | rev | cut -c3- | rev)
BASE="${BASE}/etc/profile.d/conda.sh"
echo "Will source conda base directory: ${BASE}"
source $BASE

# find the Images folder in the trial directory
find "$trial_dir" -type d -name "images" -print0 | while IFS= read -r -d $'\0' images_folder

# Perform deepfly3d on the specified trial_dir using the specified camera_ids
do
    # Define the output directory inside trial_dir
    output_path="$trial_dir/$output_dir/df3d"
    # Ensure the output directory exists
    mkdir -p "$output_path"
    
    # Run df3d-cli and check if successful
    if ! CUDA_VISIBLE_DEVICES=0 df3d-cli -vv -o "$images_folder" --output-folder "$output_path" --order $camera_ids 2>&1; then
        echo "df3d-cli command failed for trial: $trial_dir"
        exit 2
    fi
done
