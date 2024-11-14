"""
sub-module to interact with sleap for a faster and lighter version of 2D pose estimation
Please install sleap according to instructions and create a conda environment called 'sleap' to use the capabilities of this module.
https://github.com/talmolab/sleap:
conda create -y -n sleap -c conda-forge -c nvidia -c sleap -c anaconda sleap
in case this does not work, try installing from source:
https://sleap.ai/installation.html#conda-from-source 
"""

import os
import subprocess
import numpy as np
import h5py
import pandas as pd

from imabeh.run.userpaths import LOCAL_DIR, user_config


def run_sleap(trial_dir, camera_num):
    """
    run sleap shell command using os.system()
    Moves the output to the correct directory (trial_dir/behData/sleap)
    """
    # get path to the script
    imabeh_path = os.path.join(LOCAL_DIR, "..")
    script_path = os.path.join(imabeh_path, "behavior", "run_sleap.sh")
    # get path to the model
    model_path = user_config["labserver_data"] + "/sleap_models/new_model_LR/models/240719_180539.single_instance.n=802"

    # get the video directory and name
    video_dir = os.path.join(trial_dir, "behData", "images")
    video_name = "camera_" + str(camera_num) + ".mp4"
    

    # Run the shell script with subprocess
    try:
        subprocess.run(["bash", script_path, video_dir, video_name, model_path], check=True)

        # copy the outputs to the correct output folder
        output_path = os.path.join(trial_dir, "behData", "sleap")
        os.makedirs(output_path, exist_ok=True)

        pred_name = f"{video_dir}/{video_name}.predictions.slp"
        os.rename(pred_name, os.path.join(output_path, video_name.split(".")[0] + ".predictions.slp"))

        out_name = f"{video_dir}/sleap_output.h5"
        os.rename(out_name, os.path.join(output_path, "sleap_output.h5"))
        
    except subprocess.CalledProcessError as e:
        error_code = e.returncode  # Get the error code

        if error_code == 2:
            raise ValueError("The input arguments are invalid - Usage: script_path video_dir video_name model_path.")
        elif error_code == 3:
            raise FileNotFoundError("The video does not exist.")
        else:
            print("Error running sleap")



def make_sleap_df(trial_dir):
    """ convert the sleap output into a pandas dataframe compatible with the main df.
    It relativizes the data to the neck location, and also calculates joint motion energy.
    Finally it saves the dataframe to the trial directory.
    If no "neck" keypoint is found, it will not relativize the data."""

    # read the sleap output
    locations, node_names = read_sleap_output(trial_dir)
    n_samples, n_keypoints, n_dim = locations.shape
    assert n_dim == 2

    # create the dataframe
    sleap_df = pd.DataFrame(index=np.arange(n_samples))

    try:
        # get median neck location
        i_neck = next((i for i, name in enumerate(node_names) if 'neck' in name), None)
        neck_fix = np.median(locations[:,i_neck,:], axis=0)
    except:
        # if no neck is found, add raw locations (subtract 0,0)
        neck_fix = [0, 0]
            
    # get relative position to neck
    for i_k, keypoint in enumerate(node_names):
        for i_d, (d, neck_d) in enumerate(zip(["x","y"], neck_fix)):
            # add raw location to dataframe
            sleap_df[f"{keypoint}_{d}"] = locations[:,i_k, i_d] - neck_d
    
    # add motion energy
    for keypoint in node_names:
        x = sleap_df[f"{keypoint}_x"].values
        y = sleap_df[f"{keypoint}_y"].values
        sleap_df[f"{keypoint}_motionenergy"] = joint_motionenergy(x, y)

    # save
    out_path = os.path.join(trial_dir, "behData", "sleap", "sleap_df.pkl")
    sleap_df.to_pickle(out_path)



## Helper functions

def read_sleap_output(trial_dir):
    """ read the sleap output file and return the locations and node names"""

    sleap_output_file = os.path.join(trial_dir, "behData", "sleap", "sleap_output.h5")

    with h5py.File(sleap_output_file, "r") as f:
        locations = np.squeeze(f["tracks"][:].T)  # returns (N_samples, N_keypoints, N_dim)
        node_names = [n.decode() for n in f["node_names"][:]]

    _, n_keypoints, n_dim = locations.shape
    assert len(node_names) == n_keypoints

    # fill any nans with previous value 
    for i_k in range(n_keypoints):
        for i_d in range(n_dim):
            locations[:,i_k, i_d] = fill_nans_with_previous(locations[:,i_k, i_d])

    return locations, node_names



def fill_nans_with_previous(array):
    """ fill any nans in an array with the previous value"""

    if np.sum(np.isnan(array)) == 0:
        print(f"found {np.sum(np.isnan(array))} nans. will replace them with previous value")
        array = array.copy()
        if np.isnan(array[0]):
            array[0] = 0
        
        while any(np.isnan(array)):
            mask = np.isnan(array)
            indices = np.where(mask)[0]
            array[mask] = array[indices-1]
            
    return array


def joint_motionenergy(x, y, moving_average=50):
    """ calculates the frame-to-frame motion energy of a point (euclidian distance) 
    and then applies a moving average to produce a smoothed motion energy signal."""

    # shift the x and y coordinates by one frame
    x2 = np.ones_like(x)*x[0]
    x2[1:] = x[:-1]
    y2 = np.ones_like(y)*y[0]
    y2[1:] = y[:-1]

    # calculate the motion energy as the euclidean distance between the two frames
    motion_energy = np.sqrt(np.sum([np.square(x-x2), np.square(y-y2)], axis=0))

    # smooth the motion energy signal
    motion_energy_smoothed = np.convolve(motion_energy, np.ones(moving_average), 'same') / moving_average

    return motion_energy_smoothed
