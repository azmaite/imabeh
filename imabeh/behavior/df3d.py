"""
sub-module to run pose estimation usind df3d and read the output.

Includes functions to prepare a run-script for DeepFly3D,
run said script and perform post-processing with df3dPostProcessing.
"""
import os
import pickle
import numpy as np
import pandas as pd
import sys

# IMPORT FUNCTIONS FROM df3d (deepfly3d package) and df3dPostProcessing
from df3d.cli import main as df3dcli
from df3dPostProcessing.df3dPostProcessing import df3dPostProcess, df3d_skeleton

# IMPORT ALL PATHS FROM USERPATHS - DO NOT add any paths outside of this import 
from imabeh.run.userpaths import user_config, LOCAL_DIR

from imabeh.run.logmanager import LogManager
from imabeh.general.main import find_file


def run_df3d(trial_dir : str, log : LogManager):
    """run deepfly3d

    Parameters
    ----------
    trial_dir : str
        directory of the trial. should contain "behData/images" folder
        df3d will be saved within this trial folder as specified in the user_config
    log: LogManager
        log manager object to log the task status
    """

    # Get the output_dir and camera_ids from user_config as well as the images_dir
    output_dir = user_config["df3d_path"]
    camera_ids = user_config["camera_order"]
    images_dir = os.path.join(trial_dir, "behData", "images")

    # Simulate the command-line arguments
    sys.argv = [
        "df3d-cli",         # The name of the command
        "-vv",              
        "-o", images_dir,  
        "--output-folder", output_dir,
        "--order", *map(str, camera_ids)
    ]
    # Call the df3d main function to run
    # MAKE SURE YOUR .bashrc FILE HAS "export CUDA_VISIBLE_DEVICES=0" 
    # OR THE GPU WONT BE USED AND DF3D WILL BE SLOW!!!!!
    df3dcli()


def find_df3d_file(directory, type : str = 'result', most_recent=False):
    """
    This function finds the path to the output files of df3d.
    Can look for 'result' = (`df3d_result*.pkl`), 'angles' = (`joint_angles*.pkl`) or 'aligned' = (`aligned_pose*.pkl`)
    where `{cam}` is the values specified in the `camera` argument. 
    If multiple files with this name are found and most_recent = False, it throws an exception.
    otherwise, it returns the most recent file.

    Parameters
    ----------
    directory : str
        Directory in which to search.
    type : str
        Type of file to search for. Can be 'result', 'angles' or 'aligned'.
    most_recent : bool
        if True, returns the most recent file if multiple are found, by default False

    Returns
    -------
    path : str
        Path to df3d output file.

    """
    search_type = {'result': 'df3d_result_*.pkl', 'angles': 'joint_angles_*.pkl', 'aligned': 'aligned_pose_*.pkl'}
    # check that the type is correct
    if type not in search_type.keys():
        raise ValueError(f"Type must be one of {list(search_type.keys())}")
    
    # find file
    return find_file(directory,
                      search_type[type],
                      "df3d output " + type,
                      most_recent=most_recent)

def postprocess_df3d_trial(trial_dir):
    """run post-processing of deepfly3d data as defined in the df3dPostProcessing package:
    Align with reference fly template and calculate leg angles.

    Parameters
    ----------
    trial_dir : string
        directory of the trial. should contain an "images" folder at some level of hierarchy
    """
    # get the path to the (most recent) df3d result file
    pose_result = find_df3d_file(trial_dir, 'result', most_recent=True)
    pose_result_name = "df3d_result"

    # run df3d post-processing
    mydf3dPostProcess = df3dPostProcess(exp_dir=pose_result, calculate_3d=True, outlier_correction=True)
    
    # align model and save
    aligned_model = mydf3dPostProcess.align_to_template(interpolate=False, scale=True, all_body=True)
    path = pose_result.replace(pose_result_name,'aligned_pose')
    with open(path, 'wb') as f:
        pickle.dump(aligned_model, f)

    # calculate leg angles and save
    leg_angles = mydf3dPostProcess.calculate_leg_angles(save_angles=False)
    path = pose_result.replace(pose_result_name, 'joint_angles')
    with open(path, 'wb') as f:
        pickle.dump(leg_angles, f)

def get_df3d_df(trial_dir):
    """load pose estimation data into a dataframe.
    Adds columns for joint position and joint angles in the final format
    (one column per leg joint angle/position)

    Parameters
    ----------
    trial_dir : str
        base directory where pose estimation results can be found

    Returns
    -------
    beh_df_path: str
        Path to df3d dataframe containing pose estimation data
    """
    # get the path to the most recent df3d result file as well as aligned and angles files
    df3d_result = find_df3d_file(trial_dir, 'result', most_recent=True)
    df3d_angles = find_df3d_file(trial_dir, 'angles', most_recent=True)
    df3d_aligned = find_df3d_file(trial_dir, 'aligned', most_recent=True)

    # read the angles and joit positions and convert naming to final format
    # currently one dictionary per leg/region with subdictionary for each angle/position
    # new format: one dictionary per leg/region and angle/position

    # make empty dictionary to store data
    df3d_dict = {}

    # read angles file
    with open(df3d_angles, "rb") as f:
        angles = pickle.load(f)
    # get leg and angle keys
    leg_keys = [key for key in angles.keys()]
    angle_keys = [key for key in angles[leg_keys[0]].keys()]
    # add joint angles with proper names and number format to df3d_dict
    for leg in leg_keys:
        if leg == "Head": # skip head, empty (no angles)
            continue
        for angle in angle_keys:
            new_name = "angle_" + leg + "_" + angle
            new_vals = np.array(angles[leg][angle])
            df3d_dict[new_name] = new_vals

    # read the joint position file
    with open(df3d_aligned, "rb") as f:
        joints = pickle.load(f)
    # get leg keys
    leg_keys = [key for key in joints.keys()]
    # add positions proper names and number format to df3d_dict
    # for each joint, iterate through x,y,z too
    for leg in leg_keys:
        joint_keys = [key for key in joints[leg].keys()]
        for joint, (i_xyz, xyz) in zip(joint_keys, enumerate(["x", "y", "z"])):
            new_name = "joint" + leg + "_" + joint + "_" + xyz
            new_vals = np.array(joints[leg][joint]["raw_pos_aligned"][:, i_xyz])
            df3d_dict[new_name] = new_vals

    # get the abdomen positions from df3d result file
    with open(df3d_result, "rb") as f:
        pose = pickle.load(f)
    # get abdomen indeces from df3d_skeleton
    abdomen_keys = ["RStripe1", "RStripe2", "RStripe3", "LStripe1", "LStripe2", "LStripe3"]
    abdomen_inds = [df3d_skeleton.index(key) for key in abdomen_keys]
    # add abdomen positions with proper names and number format to df3d_dict
    for i_abd, abd_key, (i_xyz, xyz) in zip(abdomen_inds, abdomen_keys, enumerate(["x", "y", "z"])):
        new_name = "joint_Abd_" + abd_key + "_" + xyz
        new_vals = np.array(pose["points3d"][:, i_abd, i_xyz])
        df3d_dict[new_name] = new_vals

    # create a dataframe with all the data
    df3d_df = pd.DataFrame(df3d_dict)

    # save the dataframe in the same folder as the df3d results as a pickle file
    df_out_dir = os.path.join(os.path.dirname(df3d_result), "df3d_df.pkl")
    print(df_out_dir)
    df3d_df.to_pickle(df_out_dir)
    
    return df_out_dir