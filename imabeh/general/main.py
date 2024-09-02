"""
Available functions:
    - find_file
    - find_sync_file
    - run_shell_command
    - combine_df
"""

from pathlib import Path
import os
import subprocess
import signal
import pandas as pd
import numpy as np

from imabeh.run.userpaths import user_config
from imabeh.run.logmanager import LogManager


def find_file(directory, name, file_type, most_recent=False):
    """
    This function finds a unique file with a given name in the directory.
    If multiple files with this name are found and most_recent = False, it throws an exception.
    otherwise, it returns the most recent file.

    Parameters
    ----------
    directory : str
        Directory in which to search.
    name : str
        Name of the file.
    file_type : str
        Type of the file (for reporting errors only)
    most_recent : bool, optional
        If True, return the most recent file if multiple files are found, by default False

    Returns
    -------
    path : str
        Path to file.
    """
    file_names = list(Path(directory).rglob(name))
    if len(file_names) > 1 and not most_recent:
        raise RuntimeError(
            f"Could not identify {file_type} file unambiguously. " +
            f"Discovered {len(file_names)} {file_type} files in {directory}."
        )
    elif len(file_names) > 1 and most_recent:
        file_names = sorted(file_names, key=lambda x: x.stat().st_mtime, reverse=True)
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
    
def combine_df(trial_dir : str, new_df_path : str, log : LogManager):
    """
    This function combines two dataframes by concatenating them along the rows.
    The main dataframe is read (or created) in the trial directory as specificied by user_config.
    Saves the combined dataframe to the main dataframe path.

    Parameters
    ----------
    trial_dir : str
        Directory of the trial. 
        The main dataframe path is determined from this and user_config.
    new_df_path : str
        Path to the new dataframe to add.

    Raises
    ------
    ValueError
        If the length of the specified index_df and the fictrac output do not match
    """
    # get default main_df_path from trial_dir and user_config
    main_df_path = os.path.join(trial_dir, user_config["processed_path"], "processed_df.pkl")
    # if folder of main_df_path does not exist, create it
    if not os.path.isdir(os.path.dirname(main_df_path)):
        os.makedirs(os.path.dirname(main_df_path))

    # read the new dataframe (.pkl)
    new_df = pd.read_pickle(new_df_path)

    # check if main df exists - if so, add new df to it
    if os.path.isfile(main_df_path):
        main_df = pd.read_csv(main_df_path)
        # check that both df have the same length (number of frames)
        if len(main_df) != len(new_df):
            if np.abs(len(main_df) - len(new_df)) <=10:
                raise ValueError("Number of frames in main and new df do not match. \n"+\
                    "Main_df has {} ticks, new_df has {} lines. \n".format(len(main_df), len(new_df))+\
                    "Trial: "+ trial_dir)
        # add to main_df
        for key in list(new_df.keys()):
            # check if key aready exists in main_df. If so, log (and replace)
            if key in list(main_df.keys()):
                log.add_line_to_log("  Replacing key {} in main processing dataframe at {}".format(key, main_df_path))
            main_df[key] = new_df[key].values

    # if it doesn't exist, make the new df the main df
    # log the creation of the main dataframe
    else:
        main_df = new_df
        log.add_line_to_log("--- Created main processing dataframe at " + main_df_path)

    # save the main dataframe
    main_df.to_csv(main_df_path, index=False)



    


