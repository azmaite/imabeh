"""
utility functions to run processing using the TaskManager
"""

import os

from imabeh.run.userpaths import GLOBAL_PATHS


def read_current_user(txt_file = GLOBAL_PATHS["txt_current_user"]):
    """
    reads the supplied text file and returns the current user name.
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

    return current_user


    