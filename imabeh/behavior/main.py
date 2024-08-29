""" 
Available functions:

FUNCTIONS TO FIND BEHAVIOR SPECIFIC FILES
    - find_seven_camera_metadata_file
    - find_fictrac_file

FUNCTIONS TO LOAD DATA
    - load_fictrac
"""

import numpy as np

from imabeh.general.main import find_file


# FUNCTIONS TO FIND FILES

def find_seven_camera_metadata_file(directory):
    """
    This function finds the path to the metadata file "capture_metadata.json" 
    created by seven camera setup and returns it.
    If multiple files with this name are found, it throws an exception.

    Parameters
    ----------
    directory : str
        Directory in which to search.

    Returns
    -------
    path : str
        Path to capture metadata file.
    """
    return find_file(directory,
                      "capture_metadata.json",
                      "seven camera capture metadata")

def find_fictrac_file(directory, camera=3):
    """
    This function finds the path to the output file of fictrac of the form `camera_{cam}*.dat`, 
    where `{cam}` is the values specified in the `camera` argument. 
    If multiple files with this name are found, it throws an exception.

    Parameters
    ----------
    directory : str
        Directory in which to search.
    camera : int
        The camera used for fictrac.

    Returns
    -------
    path : str
        Path to fictrac output file.

    """
    return find_file(directory,
                      f"camera_{camera}*.dat",
                      "fictrac output")



# FUNCTIONS TO LOAD DATA

def load_fictrac(path, ball_radius=5, fps=100):
    """
    This functions loads the fictrac data from file.

    Parameters
    ----------
    path : str
        Path to fictrac output file (.dat).
    ball_radius : int
        Radius of the spherical treadmill.
    fps : float
        Number of frames per second.

    Returns
    -------
    data : dictionary
        A dictionary with the following keys:
          Speed, x, y, forward_pos, side_pos, delta_rot_lab_side, delta_rot_lab_forward, 
          delta_rot_lab_turn, integrated_forward_movement, integrated_side_movement, Time
        All speeds are in mm/s and all positions are in mm.
    """
    col_names = [
        "Frame_counter", "delta_rot_cam_right", "delta_rot_cam_down",
        "delta_rot_cam_forward", "delta_rot_error", "delta_rot_lab_side",
        "delta_rot_lab_forward", "delta_rot_lab_turn", "abs_rot_cam_right",
        "abs_rot_cam_down", "abs_rot_cam_forward", "abs_rot_lab_side",
        "abs_rot_lab_forward", "abs_rot_lab_turn", "integrated_lab_x",
        "integrated_lab_y", "integrated_lab_heading",
        "animal_movement_direction_lab", "animal_movement_speed",
        "integrated_forward_movement", "integrated_side_movement", "timestamp",
        "seq_counter", "delta_time", "alt_time"
    ]

    dat_table = np.genfromtxt(path, delimiter=",")
    data = {}
    for i, col in enumerate(col_names):
        data[col] = dat_table[:, i]
    data["Speed"] = data["animal_movement_speed"] * ball_radius * fps
    data["x"] = data["integrated_lab_x"] * ball_radius
    data["y"] = data["integrated_lab_y"] * ball_radius
    data["forward_pos"] = data["integrated_forward_movement"] * ball_radius
    data["side_pos"] = data["integrated_side_movement"] * ball_radius
    data["delta_rot_lab_side"] = data["delta_rot_lab_side"] * ball_radius * fps
    data["delta_rot_lab_forward"] = data[
        "delta_rot_lab_forward"] * ball_radius * fps
    data["delta_rot_lab_turn"] = data[
        "delta_rot_lab_turn"] / 2 / np.pi * 360 * fps
    data["integrated_forward_movement"] = data[
        "integrated_forward_movement"] * ball_radius
    data["integrated_side_movement"] = data[
        "integrated_side_movement"] * ball_radius
    data["Time"] = data["Frame_counter"] / fps

    return data
