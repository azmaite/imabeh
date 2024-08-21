"""
utility functions to run processing using the TaskManager
"""

import os
import importlib
import pandas as pd

from imabeh.run.userpaths import GLOBAL_PATHS, LOCAL_DIR


def read_current_user(txt_file = GLOBAL_PATHS["txt_current_user"]):
    """
    Reads the supplied text file and returns the current user name.
    It checks that the text file exists, that the format is correct,
    and that the user exists in the userpaths file.

    Format in the txt file:
    CURRENT_USER = USER_XXX
    Must match existing dictionary in 'run/userpaths.py'

    Parameters
    ----------
    txt_file : str, optional
        location of the text file, default set in run/userpaths.py

    Returns
    -------
    current_user_settings : dict
        user specific settings from the userpaths.py file

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
    It checks that the text file exists, that the format is correct,
    and that the fly directories within it exist.

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
    fly_dict : dict
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



## FUNCTIONS TO MANAGE THE FLY PROCESSING TABLE

def _initialize_fly_table():
    """ Create an empty fly processing table file. 
    Returns
    -------
    fly_table : pandas.DataFrame
        empty fly processing status table"""

    # get the list of possible tasks 
    from imabeh.run.tasks import task_collection

    # make a pandas dataframe with the header
    header = ["fly_dir", "trial", "pipelines"] + list(task_collection.keys()) + ["user", "comments"]
    fly_table = pd.DataFrame(columns=header)

    # save the dataframe to a csv file
    save_fly_table(fly_table)

    return fly_table

    
def _check_new_tasks(fly_table):
    """ Check that all tasks in the task.collection are in the fly processing table
    and add them if they are not.
    Parameters
    ----------
    fly_table : pandas.DataFrame
        current table with the fly processing status
    
    Returns
    -------
    fly_table : pandas.DataFrame
        updated table with new tasks added (or the original table if no new tasks were added)
    """

    # get the list of possible tasks 
    from imabeh.run.tasks import task_collection
    task_collection = list(task_collection.keys())

    # check that all tasks are in the table and add them if they are not
    new_columns = False
    for task in task_collection:
        if task not in fly_table.columns:
            fly_table[task] = 0
            new_columns = True
    
    # if new columns were added, save the table
    if new_columns:
        print('New tasks added to the fly processing table. Saving new table...')
        save_fly_table(fly_table)

    # return the updated table
    return fly_table


def get_fly_table():
    """ Get the fly status processing table.
    Check that the table exists, and create one if it does not.
    Check that all the possible tasks are in the table, and add them if they are not.

    Returns
    -------
    fly_table : pandas.DataFrame
        fly processing status table
    """
    # get the path for the fly processing table
    data_path = read_current_user()["labserver_data"]
    fly_table_path = os.path.join(data_path, GLOBAL_PATHS["csv_fly_table"])

    # check that the fly processing table file exists, and create one if not
    if not os.path.exists(fly_table_path):
        print("No fly processing status table found. Creating new one...")
        _initialize_fly_table()

    # load the fly processing table as pandas dataframe
    fly_table = pd.read_csv(fly_table_path)

    # check that all the possible tasks are in the table and add if not add them
    fly_table = _check_new_tasks(fly_table)

    return fly_table

def save_fly_table(fly_table):
    """ Save the fly processing status table to the csv file. 
    Parameters
    ----------
    fly_table : pandas.DataFrame
        table with the current fly processing status
    """
    # get the path for the fly processing table
    data_path = read_current_user()["labserver_data"]
    fly_table_path = os.path.join(data_path, GLOBAL_PATHS["csv_fly_table"])

    # save the dataframe to a csv file
    fly_table.to_csv(fly_table_path, index=False)


def _find_fly_in_fly_table(fly_table, single_trial):
    """ Find a fly to the processing status table. 

    Parameters
    ----------
    fly_table : pandas.DataFrame
        table with the current fly processing status
    single_trial : dict
        dictionary with a single fly TRIAL information (dir, trial)
    
    Returns
    -------
    fly_index: int
        row index of fly in the processing table. 
        returns -1 if the fly is not in the table
    """

    # check that the input fly_dict format is correct (only one trial)
    if not all([key in single_trial for key in ["dir", "trial"]]):
        raise ValueError("Fly trial dictionary must have 'dir' and 'trial' keys.")
    if "," in single_trial["trial"]:
        raise ValueError("Fly trial dictionary must have only one trial.")
    
    # find the index of the fly trial in the processing table
    fly_index = fly_table[
        (fly_table["fly_dir"] == single_trial["dir"]) & 
        (fly_table["trial"] == single_trial["trial"])
    ].index
    
    if len(fly_index) == 0:
        return -1
    else:
        return fly_index[0]

def _add_fly_to_fly_table(fly_table, single_trial):
    """ Add a new fly trials to the fly processing status table. 

    Parameters
    ----------
    single_trial : dict
        dictionary with a single fly TRIAL information (dir, trial)
    fly_table : pandas.DataFrame
        table with the current fly processing status
    """

    # check that the input fly_dict format is correct
    if not all([key in single_trial for key in ["dir", "trial"]]):
        raise ValueError("Single trial dictionary must have 'dir' and 'trial' keys.")

    # Check whether the fly trial is already in the table
    fly_index = _find_fly_in_fly_table(fly_table, single_trial)

    # if the fly is not in the table, add it
    if fly_index == -1:
        new_row = {
            "fly_dir": single_trial["dir"],
            "trial": single_trial['trial'],
            "pipelines": ' ',
            "user": read_current_user()["initials"],
            "comments": ' '
        }
        # Add a zero for each task in the table that isn't already in new_row (tasks)
        for column in fly_table.columns:
            if column not in new_row:
                new_row[column] = 0

        # Append the new row to the fly_table
        fly_table = fly_table.append(new_row, ignore_index=True)
    
    fly_index = _find_fly_in_fly_table(fly_table, single_trial)

    # return the updated table
    return fly_table, fly_index


def update_fly_table(fly_table, fly_trial, task_list, status_list):
    """ Update the fly processing status table for a single fly trial. 
    Parameters
    ----------
    fly_table : pandas.DataFrame
        table with the current fly processing status
    fly_trial : dict
        dictionary with the fly trial information for a SINGLE TRIAL (dir, trial).
    task_list : list
        list of tasks to update in the table
    status_list : list
        list of status to update in the table (one per task)"""

    # find the index of the fly trial in the processing table (already checks format)
    fly_index = _find_fly_in_fly_table(fly_table, fly_trial)
    if fly_index == -1:
        # if not found, add the fly to the table
        fly_table, fly_index = _add_fly_to_fly_table(fly_table, fly_trial)

    # update the table with the new status
    for task, status in zip(task_list, status_list):
        fly_table.loc[fly_index, task] = status

    return fly_table
    

## FUNCTIONS TO MANAGE THE TASK LOG

def _delete_old_logs():
    """ Delete logs older than 14 days from the logs folder
    to avoid having too many old log files in the folder.
    """

    # get the path for the log folder
    log_folder = os.path.join(LOCAL_DIR, 'logs')

    # get the list of log files in the folder
    log_files = os.listdir(log_folder)

    for log_file in log_files:
        # Get log file creation date from name
        log_date = pd.to_datetime(log_file[4:19], format='%Y%m%d_%H%M%S')
        
        # Delete if older than 14 days
        if pd.Timestamp.now() - log_date > pd.Timedelta(days=14):
            os.remove(os.path.join(log_folder, log_file))


def create_task_log():
    """ Create new task log file with the current date and time. 
    
    Returns
    -------
    task_log_path : str
        path to the new task log file
    """

    # delete old logs (to avoid having too many old log files)
    _delete_old_logs()

    # create a log folder if it doesn't exist
    task_log_folder = os.path.join(LOCAL_DIR, 'logs')
    if not os.path.exists(task_log_folder):
        os.makedirs(task_log_folder)

    # name the task log using current datetime
    now = pd.Timestamp.now()
    task_log_name = 'log_' + now.strftime("%Y%m%d_%H%M%S") + '.txt'

    # get the path for the task log
    task_log_path = os.path.join(task_log_folder, task_log_name)

    # create a new task log file
    with open(task_log_path, "w") as file:
        file.write(f"Task log created on {now}\n")
        file.write('\n')

    return task_log_path


def add_line_to_log(task_log_path, line):
    """ Add a line to the current task log file.
    
    Parameters
    ----------
    task_log_path : str
        path to the current task log file
    line : str
        line to add to the task log file
    """

    # add a new line to the task log
    with open(task_log_path, "a") as file:
        file.write(line + '\n')



# def function():
#     # get the fly table (create if it does not exist, and update task list if needed)
#     fly_table = get_fly_table()
#     # do stuff
#     ...................
#     # update the table status for the given fly trial
#     update_fly_table(fly_table, fly_trial, task_list, status_list)
#     # save the table
#     save_fly_table(fly_table)