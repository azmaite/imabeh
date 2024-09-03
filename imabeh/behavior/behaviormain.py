""" 
Available functions:

FUNCTIONS TO FIND BEHAVIOR SPECIFIC FILES
    - find_seven_camera_metadata_file
    - find_fictrac_file

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

def find_fictrac_file(directory, camera=3, most_recent=False):
    """
    This function finds the path to the output file of fictrac of the form `camera_{cam}*.dat`, 
    where `{cam}` is the values specified in the `camera` argument. 
    If multiple files with this name are found and most_recent = False, it throws an exception.
    otherwise, it returns the most recent file.

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
                      "fictrac output",
                      most_recent=most_recent)




