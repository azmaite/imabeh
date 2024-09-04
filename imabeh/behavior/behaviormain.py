""" 
Available functions:

FUNCTIONS TO FIND BEHAVIOR SPECIFIC FILES
    - find_seven_camera_metadata_file
    - find_fictrac_file

"""

import numpy as np

from imabeh.general.main import find_file


# FUNCTIONS TO FIND GENERAL FILES

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


