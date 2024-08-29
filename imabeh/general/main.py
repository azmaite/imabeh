"""
Available functions:

FUNCTIONS TO FIND GENERAL FILES
    - find_file
    - find_sync_file
"""

from pathlib import Path

def find_file(directory, name, file_type):
    """
    This function finds a unique file with a given name in the directory.
    If multiple files with this name are found, it throws an exception.

    Parameters
    ----------
    directory : str
        Directory in which to search.
    name : str
        Name of the file.
    file_type : str
        Type of the file (for reporting errors only)

    Returns
    -------
    path : str
        Path to file.
    """
    file_names = list(Path(directory).rglob(name))
    if len(file_names) > 1:
        raise RuntimeError(
            f"Could not identify {file_type} file unambiguously. " +
            f"Discovered {len(file_names)} {file_type} files in {directory}."
        )
    elif len(file_names) == 0:
        raise FileNotFoundError(f"No {file_type} file found in {directory}")
    
    return str(file_names[0])

def find_sync_file(directory):
    """
    This function finds the path to the sync file "Episode001.h5" 
    created by ThorSync and returns it.
    If multiple files with this name are found, it throws an exception.

    Parameters
    ----------
    directory : str
        Directory in which to search.

    Returns
    -------
    path : str
        Path to sync file.
    """
    return find_file(directory,
                      "Episode001.h5",
                      "synchronization")

