"""
utility functions to run processing using the TaskManager
"""

import os
import importlib

from imabeh.run.userpaths import GLOBAL_PATHS


def read_current_user(txt_file = GLOBAL_PATHS["txt_current_user"]):
    """
    Reads the supplied text file and returns the current user name.
    Format in the txt file:
    CURRENT_USER = USER_XXX
    Must match existing dictionary in 'run/userpaths.py'

    Parameters
    ----------
    txt_file : str, optional
        location of the text file, default set in run/userpaths.py

    Returns
    -------
    current_user : str

    """
    # check that the file exists
    if not os.path.exists(txt_file):
        raise FileNotFoundError(f"File {txt_file} does not exist. Please create it with the CURRENT_USER variable set.")

    # read the file
    with open(txt_file) as file:
        lines = file.readlines()
        lines = [line.rstrip() for line in lines]

    # read the file to get the current user
    for line in lines:
        if line.startswith("CURRENT_USER"):
            current_user = line.split("=")[1].strip()

    # check that the file does contain the CURRENT_USER variable
    try:
        current_user
    except NameError:
        raise ValueError(f"File does not contain the CURRENT_USER variable. Please set it to the user you want to run.")
    
    # check that the current user exists as a dictionary in the userpaths file
    try:
        exec(f"from imabeh.run.userpaths import {current_user}")
    except ImportError:
        raise ValueError(f"User {current_user} does not exist in the userpaths.py file. Please create a dictionary for this user.")

    # get the current user specific settings
    userpaths_module = importlib.import_module('imabeh.run.userpaths')
    current_user_settings = getattr(userpaths_module, current_user)

    return current_user_settings


def read_fly_dirs(txt_file = GLOBAL_PATHS["txt_file_to_process"]):
    """
    reads the supplied text file and returns a list of dictionaries
    with information for each fly to process and the tasks to run on it.
    General requested format of a line in the txt file (see example in file):
    fly_dir||trial1,trial2||task1,task2,!task3,
    ! before a task forces an overwrite.
    example:
    date_genotype/Fly1||001_beh||fictrac,!df3d

    Parameters
    ----------
    txt_file : str, optional
        location of the text file, by default set in GLOBAL_PATHS["txt_file_to_process"]

    Returns
    -------
    List[dict]
        fly dict with the following fields:
        - "dir": the base directory of the fly
        - "selected_trials": a string describing which trials to run on,
                             e.g. "001,002" or "all_trials"
        - "tasks": a comma separated string containing the names of the tasks todo

    """
    # check that the file exists
    if not os.path.exists(txt_file):
        raise FileNotFoundError(f"File {txt_file} does not exist. Please create it with the flies to process.")
    
    # read file
    with open(txt_file) as file:
        lines = file.readlines()
        lines = [line.rstrip() for line in lines]
    fly_dicts = []

    # get the flies to process
    for line in lines:
        if line.startswith("#") or line == "":
            continue
        strings = line.split("||")
        # check that the line has the correct format
        if len(strings) != 3:
            raise ValueError(f"Line {line} does not have the correct format. Please use 'fly_dir||trial1,trial2||task1,task2,!task3,'")
        # get the flies and tasks
        fly = {
            "dir": strings[0],
            "selected_trials": strings[1],
            "tasks": strings[2]
        }
        fly_dicts.append(fly)

    # Check that the fly dirs exist
    current_user_settings = read_current_user()
    data_path = current_user_settings["labserver_data"]

    for fly_dict in fly_dicts:
        trial_path = os.path.join(data_path, fly_dict["dir"])
        if not os.path.exists(trial_path):
            raise FileNotFoundError(f"Fly directory {fly_dict['dir']} does not exist. Please check fly list {txt_file}.")
        else:
            # check that the trials exist
            trials = fly_dict["selected_trials"].split(",")
            for trial in trials:
                trial_path = os.path.join(data_path, fly_dict["dir"], trial)
                if not os.path.exists(trial_path):
                    raise FileNotFoundError(f"Trial {trial} does not exist in fly directory {fly_dict['dir']}. Please check fly list {txt_file}.")

    return fly_dicts


    