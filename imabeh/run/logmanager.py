"""
Log manager module housing the LogManager class.
Manages the logs to keep track of processing status, timing and errors.

Logs will be saved within the imabeh/run/logs folder (default) as txt files,
named usign the time of creating as 'log_date_time.txt'.

The LogManager class contains the followingfunctions:
    _delete_old_logs - Delete old log files (> 14 days old)
    create_task_log - Create a new log file
    add_line_to_log - Add a line to the current log file
"""

import os
import pandas as pd

from imabeh.run.userpaths import LOCAL_DIR


class LogManager():
    """
    class to manage task execution logs
    """
    def __init__(self, log_folder: str = '') -> None:
        """
        class to create, update, and delete execution logs

        Parameters
        ----------
        log_folder : str
            path to the folder where the logs are stored
        
        Other properties
        ----------------
        log_file : str
            name of the current log file
        """
        
        # PROPERTIES
        self.log_folder = log_folder
        self.log_file = ''

        # if the log folder is not provided, use the default log folder
        if log_folder == '':
            self.log_folder = os.path.join(LOCAL_DIR, 'logs')        

        # create a log folder if it doesn't exist
        if not os.path.exists(self.log_folder):
            os.makedirs(self.log_folder)


    def _delete_old_logs(self):
        """ Delete logs older than 14 days from the logs folder
        to avoid having too many old log files in the folder.
        """

        # get the list of log files in the log folder
        log_files = os.listdir(self.log_folder)

        for log_file in log_files:
            # Get log file creation date from file name
            log_date = pd.to_datetime(log_file[4:19], format='%Y%m%d_%H%M%S')
            
            # Delete if older than 14 days
            if pd.Timestamp.now() - log_date > pd.Timedelta(days=14):
                os.remove(os.path.join(self.log_folder, log_file))


    def create_task_log(self):
        """ Create new log file with the current date and time. 
        Saves the new log file name in the task manager object.
        Also checks for and deletes old logs.
        """

        # delete old logs (to avoid having too many old log files)
        self._delete_old_logs()

        # name the task log using current datetime
        now = pd.Timestamp.now()
        self.log_file = 'log_' + now.strftime("%Y%m%d_%H%M%S") + '.txt'
        
        # create a new task log file
        log_file_path = os.path.join(self.log_folder, self.log_file)
        with open(log_file_path, "w") as file:
            file.write(f"Task log created on {now}\n")
            file.write('\n')


    def add_line_to_log(self, line: str):
        """ Add a line to the current task log file.
        
        Parameters
        ----------
        line : str
            line to add to the task log file
        """

        # add a new line to the task log
        log_file_path = os.path.join(self.log_folder, self.log_file)
        with open(log_file_path, "a") as file:
            file.write(line + '\n')
