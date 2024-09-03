"""
sub-module to run pose estimation usind df3d and read the output.

Includes functions to prepare a run-script for DeepFly3D,
run said script and perform post-processing with df3dPostProcessing.
"""
import os
from shutil import copy
import glob
import pickle
import numpy as np
import pandas as pd

from df3dPostProcessing.df3dPostProcessing import df3dPostProcess, df3d_skeleton

# IMPORT ALL PATHS FROM USERPATHS - DO NOT add any paths outside of this import 
from imabeh.run.userpaths import user_config, LOCAL_DIR

from imabeh.run.logmanager import LogManager
from imabeh.general.main import find_file


#FILE_PATH = os.path.realpath(__file__)
#BEHAVIOUR_PATH, _ = os.path.split(FILE_PATH)

#from twoppp.utils import makedirs_safe, find_file
#from twoppp import TMP_PATH


def run_df3d(trial_dir : str, log : LogManager):
    """run deepfly3d shell commands using os.system()

    Parameters
    ----------
    trial_dir : str
        directory of the trial. should contain an "images" folder at some level of hierarchy
        df3d will be saved within this folder as specified in the user_config
    log: LogManager
        log manager object to log the task status
    """

    # Get the output_dir and camera_ids from user_config
    output_dir = user_config["df3d_path"]
    camera_ids = user_config["camera_order"]
    # for the df3d folder, df3d-cli will automatically add /df3d to the end of the output_dir, so we need to remove it
    if not output_dir.endswith("/df3d"):
        raise ValueError("df3d output directory must end in '/df3d'")
    output_dir = output_dir.rstrip("/df3d")

    # go to imabeh/behavior folder to run script
    behavior_dir = os.path.join(os.path.dirname(LOCAL_DIR), 'behavior')
    os.chdir(behavior_dir)

    # Run the shell script with these variables as arguments and read the exit code
    camera_ids_str = ' '.join(map(str, camera_ids))
    error = os.system(f"bash ./run_df3d.sh {trial_dir} {output_dir} \"{camera_ids_str}\"")
    error = error >> 8 # os.system returns the exit code as a 16-bit number, so we shift it to get the actual exit code

    # log the error status(0=success,1=missing inputs, 2=df3d-cli failed)
    if error != 0:
        log.add_line_to_log(f"Error {error} while running df3d-cli (1=missing inputs, 2=df3d-cli failed)")
        raise RuntimeError(f"Error running df3d-cli - error: {error}")

def find_df3d_file(directory, most_recent=False):
    """
    This function finds the path to the output file of df3d of the form `df3d_result*.pkl`, 
    where `{cam}` is the values specified in the `camera` argument. 
    If multiple files with this name are found and most_recent = False, it throws an exception.
    otherwise, it returns the most recent file.

    Parameters
    ----------
    directory : str
        Directory in which to search.
    most_recent : bool
        if True, returns the most recent file if multiple are found, by default False

    Returns
    -------
    path : str
        Path to df3d output file.

    """
    return find_file(directory,
                      f"df3d_result_*.pkl",
                      "df3d output",
                      most_recent=most_recent)

def postprocess_df3d_trial(trial_dir):
    """run post-processing of deepfly3d data as defined in the df3dPostProcessing package:
    Align with reference fly template and calculate leg angles.

    Parameters
    ----------
    trial_dir : string
        directory of the trial. should contain an "images" folder at some level of hierarchy
    """
    # get the path to the df3d result file
    pose_result = find_df3d_file(trial_dir)
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


# def get_df3d_dataframe(trial_dir, index_df=None, out_dir=None, add_abdomen=True):
    # """load pose estimation data into a dataframe, potentially one that is synchronised
    # to the two-photon recordings.
    # Adds columns for joint position and joint angles.

    # Parameters
    # ----------
    # trial_dir : str
    #     base directory where pose estimation results can be found

    # index_df : pandas Dataframe or str, optional
    #     pandas dataframe or path of pickle containing dataframe to which the df3d result is added.
    #     This could, for example, be a dataframe that contains indices for synching with 2p data,
    #     by default None

    # out_dir : str, optional
    #     if specified, will save the dataframe as .pkl, by default None

    # add_abdomen: bool, optional
    #     if specified, search for abdominal markers in raw pose results

    # Returns
    # -------
    # beh_df: pandas DataFrame
    #     Dataframe containing behavioural data
    # """

    # if index_df is not None and isinstance(index_df, str) and os.path.isfile(index_df):
    #     index_df = pd.read_pickle(index_df)
    # if index_df is not None:
    #     assert isinstance(index_df, pd.DataFrame)
    # beh_df = index_df

    # images_dir = os.path.join(trial_dir, "images")
    # if not os.path.isdir(images_dir):
    #     images_dir = os.path.join(trial_dir, "behData", "images")
    #     if not os.path.isdir(images_dir):
    #         images_dir = find_file(trial_dir, "images", "images folder")
    #         if not os.path.isdir(images_dir):
    #             raise FileNotFoundError("Could not find 'images' folder.")
    # df3d_dir = os.path.join(images_dir, "df3d")
    # if not os.path.isdir(images_dir):
    #     df3d_dir = find_file(images_dir, "df3d", "df3d folder")
    #     if not os.path.isdir(images_dir):
    #         raise FileNotFoundError("Could not find 'df3d' folder.")

    # # read the angles and convert them into an understandable format
    # angles_file = find_file(df3d_dir, name="joint_angles*", file_type="joint angles file")
    # with open(angles_file, "rb") as f:
    #     angles = pickle.load(f)
    # leg_keys = []
    # _ = [leg_keys.append(key) for key in angles.keys()]
    # angle_keys = []
    # _ = [angle_keys.append(key) for key in angles[leg_keys[0]].keys()]

    # if "Head" in leg_keys:
    #     N_features = (len(leg_keys) - 1) * len(angle_keys)
    # else:
    #     N_features = len(leg_keys) * len(angle_keys)

    # N_samples = len(angles[leg_keys[0]][angle_keys[0]])
    # X = np.zeros((N_samples, N_features), dtype="float64")
    # X_names = []
    # for i_leg, leg in enumerate(leg_keys):
    #     if leg == "Head":
    #         continue
    #     for i_angle, angle in enumerate(angle_keys):
    #         X[:, i_angle + i_leg*len(angle_keys)] = np.array(angles[leg][angle])
    #         X_names.append("angle_" + leg + "_" + angle)

    # # read the joints from df3d after post-processing and convert them into an understandable format
    # joints_file = find_file(df3d_dir, name="aligned_pose*", file_type="aligned pose file")
    # with open(joints_file, "rb") as f:
    #     joints = pickle.load(f)
    # leg_keys = list(joints.keys())
    # joint_keys = list(joints[leg_keys[0]].keys())
    # if "Head" in leg_keys:
    #     head_keys = list(joints["Head"].keys())
    #     N_features = (len(leg_keys) - 1) * len(joint_keys) + len(head_keys)
    # else:
    #     head_keys = []
    #     N_features = len(leg_keys) * len(joint_keys)
    # Y = np.zeros((N_samples, N_features*3), dtype="float64")
    # Y_names = []
    # for i_leg, leg in enumerate(leg_keys):
    #     if leg == "Head":
    #         continue
    #     for i_joint, joint in enumerate(joint_keys):
    #         Y[:, i_leg*len(joint_keys)*3 + i_joint*3 : i_leg*len(joint_keys)*3 + (i_joint+1)*3] = \
    #             np.array(joints[leg][joint]["raw_pos_aligned"])
    #         Y_names += ["joint_" + leg + "_" + joint + i_ax for i_ax in ["_x", "_y", "_z"]]
    # if "Head" in leg_keys:
    #     N_legs = len(leg_keys) - 1
    #     for i_key, head_key in enumerate(head_keys):
    #         Y[:, N_legs*len(joint_keys)*3 + i_key*3 : N_legs*len(joint_keys)*3 + (i_key+1)*3] = \
    #                 np.array(joints["Head"][head_key]["raw_pos_aligned"])
    #         Y_names += ["joint_Head_" + head_key + i_ax for i_ax in ["_x", "_y", "_z"]]

    # if add_abdomen:
    #     try:
    #         pose_file = find_file(df3d_dir, name="df3d_result*", file_type="df3d result file")
    #     except:
    #         print("It seems like you are using an old version of DeepFly3D. Will seach for 'pose_result' file instead of 'df3d_result'")
    #         pose_file = find_file(df3d_dir, name="pose_result*", file_type="pose result file")
    #     with open(pose_file, "rb") as f:
    #         pose = pickle.load(f)
    #     abdomen_keys = ["RStripe1", "RStripe2", "RStripe3", "LStripe1", "LStripe2", "LStripe3"]
    #     abdomen_inds = [df3d_skeleton.index(key) for key in abdomen_keys]

    #     N_features = len(abdomen_keys)
    #     Z = np.zeros((N_samples, N_features*3), dtype="float64")
    #     Z_names = []
    #     for i_key, (i_abd, abd_key) in enumerate(zip(abdomen_inds, abdomen_keys)):
    #         Z[:, i_key*3:(i_key+1)*3] = pose["points3d"][:,i_abd, :]
    #         Z_names += ["joint_Abd_" + abd_key + i_ax for i_ax in ["_x", "_y", "_z"]]

    # if beh_df is None:
    #     # if no index_df was supplied externally,
    #     # try to get info from trial directory and create empty dataframe
    #     frames = np.arange(N_samples)
    #     try:
    #         fly_dir, trial = os.path.split(trial_dir)
    #         date_dir, fly = os.path.split(fly_dir)
    #         _, date_genotype = os.path.split(date_dir)
    #         date = int(date_genotype[:6])
    #         genotype = date_genotype[7:]
    #         fly = int(fly[3:])
    #         i_trial = int(trial[-3:])
    #     except:
    #         date = 123456
    #         genotype = ""
    #         fly = -1
    #         i_trial = -1
    #     indices = pd.MultiIndex.from_arrays(([date, ] * N_samples,  # e.g 210301
    #                                             [genotype, ] * N_samples,  # e.g. 'J1xCI9'
    #                                             [fly, ] * N_samples,  # e.g. 1
    #                                             [i_trial, ] * N_samples,  # e.g. 1
    #                                             frames
    #                                         ),
    #                                         names=[u'Date', u'Genotype', u'Fly', u'Trial',u'Frame'])
    #     beh_df = pd.DataFrame(index=indices)

    # beh_df[X_names] = X
    # beh_df[Y_names] = Y
    # if add_abdomen:
    #     beh_df[Z_names] = Z

    # if out_dir is not None:
    #     beh_df.to_pickle(out_dir)

    # return beh_df