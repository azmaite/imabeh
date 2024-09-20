"""
sub-module to define paths and user settings for running pre-processing with.
Also contains a function to read the current user from a text file and combine 
the default settings with the user specific settings.

To adapt the parameters for your use:
1. create a dictionary for your user and define the paths and parameters accordingly
2. check that the paths in the dictionaries are correct for your workstation
3. set CURRENT_USER at the top of the "_user_and_fly_dirs_to_process.txt" file equal to the dictionary you just created



also see README.md in the imabeh/run folder for run instructions
"""
import os
import importlib

LOCAL_DIR, _ = os.path.split(os.path.realpath(__file__))

# Global paths that are the same for all users
GLOBAL_PATHS = {
    # location of the labserver data path
    "labserver_data": "/mnt/upramdya_data",

    # location of the txt file that specifies the current user and the list of fly_dirs to process
    # This cannot be changed user by user!
    "txt_user_and_dirs_to_process": os.path.join(LOCAL_DIR, "_user_and_fly_dirs_to_process.txt"),
}

DEFAULT_PATHS = {
    # relative paths to different folders within each trial (to append to each trial path)
    "2p_path": "2p",
    "septacam_path": "behData/images",
    "fictrac_path": "behData/fictrac",
    "df3d_path": "behData/df3d", # MUST END in /df3d! Otherwise, df3d-cli won't work
    "processed_path": "processed",
    "figures_path": "figures",

    # environment names for different processing steps
    "sleap_env": "sleap",
    
    # name of csv file to keep track of which fly folders have been processed
    "csv_fly_table": "_fly_processing_status.csv",
    # location of file with currently running tasks
    "txt_file_running": os.path.join(LOCAL_DIR, "_tasks_running.txt"),
}

# settins for cameras of each scope
SCOPE_CONFIG = {
    "2p_1": {
        "camera_order": [0, 1, 2, 3, 4, 5, 6], # order of cameras for df3d
        "fictrac_cam": 4, # camera used for fictrac
        "opto_cam": 2     # camera with clearest laser stimulation spot
    },
    "2p_2": {
        "camera_order": [6, 5, 4, 3, 2, 1, 0],
        "fictrac_cam": 3,
        "opto_cam": 5,
    },
    "scape": {
        "camera_order": [6, 5, 4, 3, 2, 1, 0],   ################################## TODO: check this
        "fictrac_cam": 3,
        "opto_cam": 5
    },
}


# default settings for each user (create a new one per person and scope)
# the default paths/settings for all users are defined in DEFAULT_PATHS and SCOPE_CONFIG
# to use any non-default paths or settings, add them to your user dictionary:
# they will have priority over the defaults

USER_EXAMPLE_2p2 = {
    "initials": "EX_2p1",
    "labserver_data": os.path.join(GLOBAL_PATHS['labserver_data'], "EX"),

    # which scope you're using
    "scope": "2p_2",

    # non-default paths or settings
    "csv_fly_table": "_example_project_status.csv",
    "opto_cam": 2,
}

# user for testing purposes
USER_TEST = {
    "initials": "TEST",
    "labserver_data": os.path.join(os.path.dirname(LOCAL_DIR), "tests"),
    "scope": "2p_2",
}

USER_MA_2p2 = {
    "initials": "MA_2p2",
    "labserver_data": os.path.join(GLOBAL_PATHS['labserver_data'], "MA"),
    "scope": "2p_2",
}

USER_MA_scape = {
    "initials": "MA_scape",
    "labserver_data": os.path.join(GLOBAL_PATHS['labserver_data'], "MA"),
    "scope": "scape",
}


def get_current_user_config(txt_file = GLOBAL_PATHS["txt_user_and_dirs_to_process"]):
    """
    Reads the supplied text file and returns the current user name,
    as well as the user specific paths/settings from the userpaths.py file.
    It then combines the user specific settings with the default settings,
    prioritizing the user specific settings.

    Format in the txt file (anywhere in the file):
    CURRENT_USER = USER_XXX
    Must match existing dictionary in 'run/userpaths.py'

    Parameters
    ----------
    txt_file : str, optional
        location of the text file, default set in run/userpaths.py

    Returns
    -------
    user_config : dict
        default + scope + user specific settings from the userpaths.py file

    """
    # read the current user from the text file (and check the format etc.)
    current_user = _read_user_from_file(txt_file)

    # get the current user specific settings
    userpaths_module = importlib.import_module('imabeh.run.userpaths')
    user_settings = getattr(userpaths_module, current_user)

    # Combine the global paths with the default paths and the user scope settings
    user_config = GLOBAL_PATHS.copy()
    user_config.update(DEFAULT_PATHS)
    user_config.update(SCOPE_CONFIG[user_settings["scope"]])
    # combine the prior settings with the user specific settings (prioritizing the user settings)
    user_config.update(user_settings)

    return user_config

def _read_user_from_file(txt_file):
    """
    Reads the supplied text file and returns the current user name.
    It checks that the text file exists, that the format is correct,
    and that the user exists in the userpaths file.

    Format in the txt file (anywhere in the file):
    CURRENT_USER = USER_XXX
    Must match existing dictionary in 'run/userpaths.py'

    Parameters
    ----------
    txt_file : str
        location of the text file

    Returns
    -------
    current_user : str
        name of the current user
    """

    # check that the file exists
    if not os.path.exists(txt_file):
        raise FileNotFoundError(f"File {txt_file} does not exist. Please create it with the CURRENT_USER variable set.")

    # read the file
    with open(txt_file) as file:
        lines = file.readlines()
        lines = [line.rstrip() for line in lines]

    # read the file to get the current user (first match)
    for line in lines:
        if line.startswith("CURRENT_USER"):
            current_user = line.split("=")[1].strip()
            break

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

    return current_user

# GET THE CURRENT USER CONFIG!!!
# This is the only thing that should be called from outside this file as follows:
# from imabeh.run.userpaths import user_config
user_config = get_current_user_config()