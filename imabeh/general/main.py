"""
Available functions:

FUNCTIONS TO FIND GENERAL FILES
    - find_file
    - find_sync_file
    - run_shell_command
"""

from pathlib import Path
import subprocess
import signal

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

def run_shell_command(command, allow_ctrl_c=True, suppress_output=False) -> bool:
    """use the subprocess module to run a shell command

    Parameters
    ----------
    command : str
        shell command to execute

    allow_ctrl_c : bool, optional
        whether a CTRL+C event will allow to continue or not, by default True

    suppress_output : bool, optional
        whether to not show outputs, by default False

    Returns
    -------
    bool
        True if the command was executed successfully, False otherwise
    """   
    try:
        if suppress_output:
            process = subprocess.Popen(command, shell=True, stdout=subprocess.DEVNULL)
        else:
            process = subprocess.Popen(command, shell=True)

        # Communicate with the process and wait for it to complete
        process.communicate()

    except KeyboardInterrupt:
        # If a KeyboardInterrupt is caught, send the interrupt signal to the process
        process.send_signal(signal.SIGINT)
        return False

    # Check the return code of the process - 0 means success
    if process.returncode == 0:
        return True
    else:
        return False