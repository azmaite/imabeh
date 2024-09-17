"""
Available functions:
    - find_file
    - find_sync_file
    - find_seven_camera_metadata_file
    - run_shell_command
    - get_sync_df
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
from imabeh.imaging2p import utils2p
from imabeh.general import syncronization as sync


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

def find_sync_metadata_file(directory):
    """
    This function finds the path to the synchronization
    metadata file "ThorRealTimeDataSettings.xml" created
    by ThorSync. If multiple files with this name are found,
    it throws an exception.

    Parameters
    ----------
    directory : str
        Directory in which to search.

    Returns
    -------
    path : str
        Path to synchronization metadata file.

    Examples
    --------
    >>> import utils2p
    >>> utils2p.find_sync_metadata_file("data/mouse_kidney_raw")
    'data/mouse_kidney_raw/2p/Sync-025/ThorRealTimeDataSettings.xml'

    """
    return find_file(directory,
                      "ThorRealTimeDataSettings.xml",
                      "synchronization metadata"
                      )

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

def run_shell_command(command, suppress_output=False) -> bool:
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

def get_sync_df(trial_dir):
    """
    Function to generate an empty processed dataframe from the Thorsync data.
    Including the frame times for two photon and behavioural data,
    as well as the optogenetic stimulation timeseries.
    Fictrac, df3d, and imaging processed data can be later added.
    """
    # get the sync files etc
    # if behavior/imaging are not present, some files will not be found
    sync_file = find_sync_file(trial_dir) # any
    sync_metadata_file = find_sync_metadata_file(trial_dir) # any
    try:
        septacam_metadata_file = find_seven_camera_metadata_file(trial_dir) # behavior only
    except FileNotFoundError:
        septacam_metadata_file = None
    try:    
        metadata_2p_file = utils2p.find_metadata_file(trial_dir) # imaging only
    except FileNotFoundError:
        metadata_2p_file = None

    # process lines
    processed_lines = sync.get_processed_lines(sync_file, sync_metadata_file, septacam_metadata_file, metadata_2p_file)

    # add the processed lines to the dataframe
    # make empty dataframe
    sync_df = pd.DataFrame()
    # add general info from the trial_dir as attributes
    sync_df = _add_attributes_to_df(sync_df, trial_dir)

    # if behavior - add frames and times:
    if septacam_metadata_file is not None:
        # get indeces of rising edges of the cameras (including first non-nan value)
        cam = processed_lines['Cameras']
        cam_idx = np.append(np.where(~np.isnan(cam))[0][0], sync.edges(cam)[0])
        # get times
        cam_times = processed_lines['Times'][cam_idx]
        # add to dataframe
        sync_df["Septacam_frames"] = cam[cam_idx].astype(int)
        sync_df["Time"] = cam_times


        # if ALSO imaging - add 2p frames in relation to behavior frames:
        if metadata_2p_file is not None:
            scope = processed_lines['Frame Counter']
            scope_idx = scope[cam_idx]
            sync_df["2p_frames"] = scope_idx.astype(int)

    # get default main_df_path from trial_dir and user_config
    # check that dir exists, if not create it
    main_df_path = os.path.join(trial_dir, user_config["processed_path"])
    if not os.path.isdir(main_df_path):
        os.makedirs(main_df_path)
    main_df_file = os.path.join(main_df_path, "processed_df.pkl")
    # save the dataframe
    sync_df.to_pickle(main_df_file)

    return sync_df


def _add_attributes_to_df(df, trial_dir):
    """ 
    Function to add general info from the trial to the dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe to add attributes to.
    trial_dir : str
        Directory of the trial.
    
    Returns
    -------
    df : pd.DataFrame
        Dataframe with added attributes
    """

    # trial_dir format = A/B/long_path/YYMMDD_GENOTYPE_GENOTYPE/FLY/TRIAL_TRIAL
    path_split = trial_dir.split('/')
    df.attrs["Trial"] = path_split[-1]
    df.attrs["Fly"] = path_split[-2]
    df.attrs["Genotype"] = path_split[-3][7:]

    # convert date format from YYMMDD to YYYY-MM-DD
    date = path_split[-3][:6]
    df.attrs["Date"] = "20" + date[:2] + "-" + date[2:4] + "-" + date[4:]

    # add the scope used as an atribute too
    df.attrs["Scope"] = user_config["scope"]

    return df


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

    # check if main df exists - if not, create use using get_sync_df
    if not os.path.isfile(main_df_path):
        main_df = get_sync_df(trial_dir)
        # log that the main dataframe was created
        log.add_line_to_log("  Main processing dataframe created at {}".format(main_df_path))
    else:
        main_df = pd.read_csv(main_df_path)
    

    # check that both df have the same length (number of frames)
    if len(main_df) != len(new_df):
        if np.abs(len(main_df) - len(new_df)) <=10:
            raise ValueError("Number of frames in main and new df do not match. \n"+\
                "Main_df has {} ticks, new_df has {} lines. \n".format(len(main_df), len(new_df))+\
                "Trial: "+ trial_dir)
    # add new to main_df
    for key in list(new_df.keys()):
        # check if key aready exists in main_df. If so, log (and replace)
        if key in list(main_df.keys()):
            log.add_line_to_log("  Replacing key {} in main processing dataframe at {}".format(key, main_df_path))
        main_df[key] = new_df[key].values

    # combine atributes too, if any!
    for key in list(new_df.attrs.keys()):
        main_df.attrs[key] = new_df.attrs[key]

    # save the main dataframe
    main_df.to_pickle(main_df_path)

def read_main_df(trial_dir):
    # get default main_df_path from trial_dir and user_config
    main_df_path = os.path.join(trial_dir, user_config["processed_path"], "processed_df.pkl")
    # read the new dataframe (.pkl)
    main_df = pd.read_pickle(main_df_path)

    return main_df




    


