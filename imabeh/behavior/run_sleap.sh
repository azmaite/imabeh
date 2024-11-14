#!/bin/bash
set -e  # Exit immediately if any command fails

# Check if the number of arguments passed is correct
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 video_dir video_name model_path"
    exit 2
fi

# Assign arguments to variables
video_dir=$1
video_name=$2
model_path=$3

# make the conda command known to the current shell by performing source /.../conda.sh
BASE=$(conda info | grep -i 'base environment' | cut -d':' -f 2 | cut -d'(' -f 1 | cut -c2- | rev | cut -c3- | rev)
BASE="${BASE}/etc/profile.d/conda.sh"
echo "  will source conda base directory: ${BASE}"
source $BASE

# activate the sleap environment
OLD_ENV=$CONDA_DEFAULT_ENV
echo "  old environment: ${OLD_ENV}"
conda deactivate
echo "  will activate 'sleap' environment"
conda activate sleap

# Run sleap on video
video_path="${video_dir}/${video_name}"
# if it fails, it will raise an error 1
if test -f "$video_path"; then
  echo "  will run sleap on $video_path."
  sleap-track $video_path -m "$model_path" --verbosity rich --batch_size 32
  sleap-convert --format analysis -o "${video_dir}/sleap_output.h5" "${video_path}.predictions.slp"
# if video doesn't exist, raise error 2
else
  echo "$video_path does not exist"
  exit 3
fi

# deactivate the sleap environment and return to previous env
conda deactivate
conda activate $OLD_ENV
