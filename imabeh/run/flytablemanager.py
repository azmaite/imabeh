"""
Status table manager module housing the FlyTableManager class.

Contains functions to:
- Create a new fly processing table
- Save the fly processing table as a csv file
- Get the fly processing table from a path
- Check that all available tasks are in the table (and add them if they are not)
- Find a fly trial in the fly processing table
- Add a new fly trial to the fly processing table
- Check the status of a fly trial and task in the fly processing table
- Update the status of a fly trial and task in the fly processing table

Will log the creation of a new table and the addition of new tasks to the table using LogManager.
"""

import os
import pandas as pd

from imabeh.run.userpaths import LOCAL_DIR, read_current_user
from imabeh.run.logmanager import LogManager


# get the current user configuration (paths and settings)
user_config = read_current_user()


class FlyTableManager():
    """
    class to manage the table of task statuses for each fly processing.
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
        self.table = None

        # if the table folder and file are not provided, use the defaults
        if table_folder == '':
            self.table_folder = user_config['labserver_data']
        if table_file == '':
            self.table_file = user_config['csv_fly_table']

        # if the table file does not exist, create a new one
        if not os.path.exists(os.path.join(self.table_folder, self.table_file)):
            self.table = self._create_fly_table()   
        else:
            # otherwise, get the existing one
            self.table = self.get_fly_table() 



    def _create_fly_table(self, log: LogManager = None):
        """ Create an empty fly processing table file and save it to the csv file.
        Also return the table as self.table.

        Parameters
        ----------
        self.table_folder : str
        self.table_file : str
        log : LogManager, optional
            If a log is provided, log the creation of the new table.
        
        Returns
        -------
        self.fly_table : pandas.DataFrame
        """

        # get the list of possible tasks 
        from imabeh.run.tasks_2 import task_collection

        # make a pandas dataframe with the header
        header = ["fly_dir", "trial", "pipelines"] + list(task_collection.keys()) + ["user", "comments"]
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
        """

        # get the path for the fly processing table
        table_path = os.path.join(self.table_folder, self.table_file)

        # save the dataframe to a csv file
        self.table.to_csv(table_path, index=False)


    def _check_new_tasks(self, log: LogManager = None):
        """ Check that all tasks in the task.collection are in the fly processing table
        and add them if they are not.
        If a log is provided, log the addition of new tasks if necessary.
        Parameters
        ----------
        self.table : pandas.DataFrame
        log : LogManager, optional
        """

        # get the list of possible tasks 
        from imabeh.run.tasks_2 import task_collection
        task_collection = list(task_collection.keys())

        # check that all tasks are in the table and add them if they are not
        new_columns = False
        for task in task_collection:
            if task not in self.table.columns:
                self.table[task] = 0
                new_columns = True
        
        # if new columns were added, save the table
        if new_columns:
            self.save_fly_table()
            # if log is provided, log the addition of new tasks
            if log is not None:
                log.add_line("STATUS_TABLE: Newly availabe tasks added to the fly processing table")


    def _get_fly_table(self):
        """ Get the fly status processing table from the path.
        Check that the table exists, and create one if it does not.
        Check that all the possible tasks are in the table, and add them if they are not.

        Parameters
        ----------
        self.table_folder : str
        self.table_file : str

        Returns
        -------
        self.table : pandas.DataFrame
        """

        # get the path for the fly processing table
        table_path = os.path.join(self.table_folder, self.table_file)

        # check that the fly processing table file exists, and create one if not
        if not os.path.exists(table_path):
            self._create_fly_table()

        # otherwise read the table from the file and check that all tasks are in the table
        else:
            self.table = pd.read_csv(table_path)
            self._check_new_tasks(self.table)    


    def _find_fly_in_fly_table(self, single_trial: dict):
        """ Find a fly in the processing status table. 

        Parameters
        ----------
        self.table : pandas.DataFrame
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
        fly_index = self.table[
            (self.table["fly_dir"] == single_trial["dir"]) & 
            (self.table["trial"] == single_trial["trial"])
        ].index
        
        if len(fly_index) == 0:
            return -1
        else:
            return fly_index[0]


    def _add_fly_to_fly_table(self, single_trial: dict):
        """ Add a new fly trials to the fly processing status table. 

        Parameters
        ----------
        self.table : pandas.DataFrame
        single_trial : dict
            dictionary with a single fly TRIAL information (dir, trial)
        """

        # Check whether the fly trial is already in the table
        # this will check the format of the input dictionary
        fly_index = self._find_fly_in_fly_table(self, single_trial)

        # if the fly is not in the table, add it
        if fly_index == -1:
            new_row = {
                "fly_dir": single_trial["dir"],
                "trial": single_trial['trial'],
                "pipelines": ' ',
                "user": user_config["initials"],
                "comments": ' '
            }
            # Add a zero for each task in the table that isn't already in new_row (tasks)
            for column in self.table.columns:
                if column not in new_row:
                    new_row[column] = 0

            # Append the new row to the fly table
            self.table = self.table.append(new_row, ignore_index=True)
        
            # get new index
            fly_index = self._find_fly_in_fly_table(self, single_trial)
        
            # save the table
            self._save_fly_table()

        # return the fly index
        return fly_index
    

    def check_trial_task_status(self, single_trial: dict, task: str):
        """ Check the status of a fly trial and task in the fly processing table. 

        Parameters
        ----------
        self.table : pandas.DataFrame
        single_trial : dict
            dictionary with a single fly TRIAL information (dir, trial)
        task : str
            task name
        
        Returns
        -------
        status : int
            status of the task for the fly trial (0 = not done, 1 = done, 2 = error)
        """

        # find the fly index in the table. Checks format already
        fly_index = self._find_fly_in_fly_table(single_trial)

        # get the status of the task for the fly trial
        status = self.table.loc[fly_index, task]

        return status


    def update_trial_task_status(self, single_trial: dict, task: str, status: int):
        """ Update the status of a fly trial and task in the fly processing table.
        Saves the table.

        Parameters
        ----------
        self.table : pandas.DataFrame
        single_trial : dict
            dictionary with a single fly TRIAL information (dir, trial)
        task : str
            task name
        status : int
            status of the task for the fly trial (0 = not done, 1 = done, 2 = error)
        """

        # find the fly index in the table. Checks format already
        fly_index = self._find_fly_in_fly_table(single_trial)

        # update the status of the task for the fly trial
        self.table.loc[fly_index, task] = status

        # save the table
        self._save_fly_table()
