"""
Status table manager module housing the FlyTableManager class.
Manages the table of processing table for each fly trial.

The fly processing table is a pandas dataframe saved as a csv that contains:
- the list of fly trials that have been analyzed (directories made of fly_dir and path)
- which tasks have been processed for each trial and their status (0 = not run, 1 - ran successfuly, 
    2 = ran with errors).
- which user analyzed the trial
- which pipeline(s) was run on the trial if any
- comments (if any)

The FlyTableManager class contains the following functions:
    check_trial_task_status - Check the status of a given fly trial and task in the fly processing table
    update_trial_task_status - Update the status of a given fly trial and task in the fly processing table, 
        adding them if necessary

Contains (private) functions:
    _create_fly_table - Create a new fly processing table
    _save_fly_table - Save the fly processing table as a csv file
    _get_fly_table - Get the fly processing table from a path
    _add_fly_to_fly_table - Add a new fly trial to the fly processing table
    _find_fly_in_fly_table - Find a fly trial in the fly processing table
    _add_new_task - Add a new task to the fly processing table

Will optionally log the creation of a new table and the addition of new tasks using LogManager.
"""

import os
import pandas as pd

from imabeh.run.userpaths import LOCAL_DIR, user_config # get the current user configuration (paths and settings)
from imabeh.run.logmanager import LogManager


class FlyTableManager():
    """
    class to manage the fly processing table.
    """
    def __init__(self, table_folder: str = '', table_file: str = '') -> None:
        """
        class to create, update and check tables to keep track of task statuses

        Parameters
        ----------
        table_folder : str
            path to the folder where the tables are stored
        table_file : str    
            file of the table file
        
        Other properties
        ----------------
        table : pandas.DataFrame
            table with the fly processing status
        """

        # PROPERTIES
        self.table_folder = table_folder
        self.table_file = table_file
        self.fly_table = None

        # if the table folder and file are not provided, use the defaults
        if table_folder == '':
            self.table_folder = user_config['labserver_data']
        if table_file == '':
            self.table_file = user_config['csv_fly_table']

        # if the table file does not exist, create a new one
        if not os.path.exists(os.path.join(self.table_folder, self.table_file)):
            self._create_fly_table()   
        else:
            # otherwise, get the existing one
            self._get_fly_table() 



    def _create_fly_table(self, log: LogManager = None):
        """ Create an empty fly processing table file and save it to the csv file.
        Also return the table as self.fly_table.

        Parameters
        ----------
        self.table_folder : str
        self.table_file : str
        log : LogManager, optional
            If a log is provided, log the creation of the new table.
        
        Generates
        -------
        self.fly_table : pandas.DataFrame
        """

        # make a pandas dataframe with the header
        header = ["fly_dir", "trial", "pipelines", "user", "comments"]
        self.fly_table = pd.DataFrame(columns=header)

        # save the dataframe to a csv file
        self._save_fly_table()

        # if a log is provided, log the creation of the new table
        if log is not None:
            log.add_line("STATUS_TABLE: New fly processing status table created")


    def _save_fly_table(self):
        """ Save the fly processing status table to the csv file path. 
        Parameters
        ----------
        self.table_folder : str
        self.table_file : str
        self.fly_table : pandas.DataFrame
        """

        # get the path for the fly processing table
        table_path = os.path.join(self.table_folder, self.table_file)

        # save the dataframe to a csv file
        self.fly_table.to_csv(table_path, index=False)


    def _get_fly_table(self):
        """ Get the fly status processing table from the path.
        Check that the table exists, and create one if it does not.

        Parameters
        ----------
        self.table_folder : str
        self.table_file : str

        Generates
        -------
        self.fly_table : pandas.DataFrame
        """

        # get the path for the fly processing table
        table_path = os.path.join(self.table_folder, self.table_file)

        # check that the fly processing table file exists, and create one if not
        if not os.path.exists(table_path):
            self._create_fly_table()

        # otherwise read the table from the file
        else:
            self.fly_table = pd.read_csv(table_path)


    def _add_fly_to_fly_table(self, single_trial: dict):
        """ Add a new fly trials to the fly processing status table. 

        Parameters
        ----------
        self.fly_table : pandas.DataFrame
        single_trial : dict
            dictionary with a single fly TRIAL information (fly_dir, trial)

        Returns
        ----------
        fly_index : int
            row index of new fly in the processing table. 
        """

        # check that the input single_trial format is correct (only one trial)
        if not all([key in single_trial for key in ["fly_dir", "trial"]]):
            raise ValueError("Fly trial dictionary must have 'fly_dir' and 'trial' keys.")
        if "," in single_trial["trial"]:
            raise ValueError("Fly trial dictionary must have only one trial.")
        
        # check that the fly is not already in the table
        if len(self.fly_table[
            (self.fly_table["fly_dir"] == single_trial["fly_dir"]) & 
            (self.fly_table["trial"] == single_trial["trial"])
            ]) > 0:
            raise ValueError("Fly trial already in the processing table.")

        # create the fly row and fill the non-task columns
        new_row = {
            "fly_dir": single_trial["fly_dir"],
            "trial": single_trial['trial'],
            "pipelines": ' ',
            "user": user_config["initials"],
            "comments": ' '
        }
        # Add a zero for each column in the table which isn't already in new_row (tasks)
        for column in self.fly_table.columns:
            if column not in new_row:
                new_row[column] = 0

        # Append the new row to the fly table
        self.fly_table = pd.concat([self.fly_table, pd.DataFrame([new_row])], ignore_index=True)

        # save the table
        self._save_fly_table()
    
        # get new index to return
        fly_index = self._find_fly_in_fly_table(single_trial)
        return fly_index


    def _find_fly_in_fly_table(self, single_trial: dict):
        """ Find a fly in the processing status table. 
        If the fly doesn't exist, add it to the table.
        First checks that input format is correct

        Parameters
        ----------
        self.fly_table : pandas.DataFrame
        single_trial : dict
            dictionary with a single fly TRIAL information (fly_dir, trial)
        
        Returns
        -------
        fly_index: int
            row index of fly in the processing table. 
        """

        # check that the input single_trial format is correct (only one trial)
        if not all([key in single_trial for key in ["fly_dir", "trial"]]):
            raise ValueError("Fly trial dictionary must have 'fly_dir' and 'trial' keys.")
        if "," in single_trial["trial"]:
            raise ValueError("Fly trial dictionary must have only one trial.")

        # find the index of the fly trial in the processing table
        fly_index = self.fly_table[
            (self.fly_table["fly_dir"] == single_trial["fly_dir"]) & 
            (self.fly_table["trial"] == single_trial["trial"])
        ].index
        
        # if the fly is not found, add it. This will return the new index.
        if len(fly_index) == 0:
            fly_index = self._add_fly_to_fly_table(single_trial)
        else:
            fly_index = fly_index[0]
        
        return fly_index
    

    def _add_new_task(self, task: str, log: LogManager = None):
        """ Add a new task to the fly processing table
        If a log is provided, log the addition of the new task.

        Parameters
        ----------
        self.fly_table : pandas.DataFrame
        task : str
            name of task to add to table
        log : LogManager, optional
        """

        # add new task as empty column
        self.fly_table[task] = 0
        
        # if log is provided, log the addition of new task
        if log is not None:
            log.add_line_to_log(f"STATUS_TABLE: New task added to the fly processing table : {task}")


    def check_trial_task_status(self, single_trial: dict):
        """ Check the status of a fly trial and task in the fly processing table. 
        If the task doesn't exist, return 0 (not done)

        Parameters
        ----------
        self.fly_table : pandas.DataFrame
        single_trial : dict
            dictionary with a single fly TRIAL+TASK information (fly_dir, trial, task)
        
        Returns
        -------
        status : int
            status of the task for the fly trial (0 = not done, 1 = done, 2 = error)
        """

        # find the fly index in the table. Checks format already
        fly_index = self._find_fly_in_fly_table(single_trial)

        # check if the task exists in the table, if not return 0 (not done)
        if single_trial['task'] not in self.fly_table.columns:
            status = 0
        else:
            # get the status of the task for the fly trial
            status = self.fly_table.loc[fly_index, single_trial['task']]

        return status


    def update_trial_task_status(self, single_trial: dict, status: int, log: LogManager = None):
        """ Update the status of a fly trial and task in the fly processing table.
        If the fly or task don't yet exist in the table, add them.
        Saves the table.

        Parameters
        ----------
        self.fly_table : pandas.DataFrame
        single_trial : dict
            dictionary with a single fly TRIAL+TASK information (fly_dir, trial, task)
        status : int
            status of the task for the fly trial (0 = not done, 1 = done, 2 = error)
        log : LogManager, optional
        """

        # find the fly index in the table. 
        # Checks format already and adds fly if not in table
        fly_index = self._find_fly_in_fly_table(single_trial)

        # check if the task exists in the table, and if not, add it
        if single_trial['task'] not in self.fly_table.columns:
            self._add_new_task(single_trial['task'], log)

        # update the status of the task for the fly trial
        self.fly_table.loc[fly_index, single_trial['task']] = status

        # save the table
        self._save_fly_table()
