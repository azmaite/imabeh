#task_collection

"""
sub-module to define different steps of pre-processing as sub-classes of Task.
List of all available tasks is defined in task_collection at the bottom (automatically generated).

Each task should have the following methods:
    __init__(self) : to define the task name and prerequisites
    _run(self, torun_dict, log) : to run the task

On top of this, the general Task class has the following methods:
    start_run(self, torun_dict, log) : to start the running of the task, by:
        - logging the start of the task in the general log
        - logging that the task is running in the taskstatus log file
        - calling the _run method of the task
        - logging the end of the task in the taskstatus log file (whether successful or not)
        - returning the path to the taskstatus log file
    test_finished(self, torun_dict) : to check if the task has finished by reading 
        the taskstatus log file (created by start_run method)
"""

# general imports
import os
import time
from datetime import datetime

# imports for the task manager to run
from imabeh.run.userpaths import LOCAL_DIR, user_config # get the current user configuration (paths and settings)
from imabeh.run.logmanager import LogManager
from imabeh.general.main import combine_df

# task specific imports
from imabeh.imaging.utils2p import create_tiffs
from imabeh.behavior.fictrac import config_and_run_fictrac, get_fictrac_df


class Task:
    """
    Base class to implement a particular processing step.
    """
    def __init__(self) -> None:
        """
        Base class to implement a particular pre-processing step.
        """
        self.name = ""
        self.params = None
        self.prerequisites = []
    
    def full_path(self, torun_dict):
        """ get full path of trial, using user_config.labserver_data """
        return os.path.join(user_config["labserver_data"], torun_dict['fly_dir'], torun_dict['trial'])
    

    def start_run(self, torun_dict : dict, log : LogManager) -> str:
        """
        method to be start the running of each Task subclass correctly, loggin it's status
        each Task subclass will need it's own run method to actually run the task

        Parameters
        ----------
        torun_dict : dict
            - fly_dir: str, the base directory of the fly, where the data is stored
            - trial: trial to run the task on
            - overwrite: bool, whether or not to force an overwrite of the previous results
            - status: bool, whether the task is "ready", "running", or "waiting" - should always be "running" for this method
        log : LogManager
            the log manager object to log the task status

        Returns
        -------
        str
            the path to the taskstatus log file
        """

        # get trial path from torun_dict
        trial_path = os.path.join(torun_dict['fly_dir'], torun_dict['trial'])
        
        # log the start of the task in general log
        log.add_line_to_log(f"Starting {self.name} task for trial {trial_path} @ {datetime.now().isoformat(sep=' ')}")

        # log that the task is running into the taskstatus log file (create a new log)
        task_log = LogManager(log_name = f"_task_{self.name}_status")
        task_log.add_line_to_log("running started at " + datetime.now().isoformat(sep=' '))

        try:
            # RUN TASK!!!
            self._run(torun_dict, log)

            # log the correct end of the task
            task_log.add_line_to_log("finished successfully at " + time.ctime(time.time()))

        except Exception as e:
            # log the failure of the task
            task_log.add_line_to_log("failed at " + time.ctime(time.time()))
            task_log.add_line_to_log(f"  Error: {e}")

        # return the path to the taskstatus log file
        return os.path.join(task_log.log_folder,task_log.log_file)


    def _run(self, torun_dict) -> bool:
        """
        abstract method to be re-used in each Task subclass to actually run the task
        """
        raise NotImplementedError("This method should be implemented in the subclass")
    

    def test_finished(self, torun_dict: dict) -> int:
        """
        abstract method to check if the task has finished by reading the taskstatus_log file

        Parameters
        ----------
        torun_dict : dict
            - fly_dir: str, the base directory of the fly, where the data is stored
            - trial: trial to run the task on
            - overwrite: bool, whether or not to force an overwrite of the previous results
            - status: bool, whether the task is "ready", "running", or "waiting" - should always be "running" for this method
            - taskstatus_log: str, the path to the taskstatus log file

        Returns
        -------
        int
            0 = still running, 1 = finished sucessfully, 2 = failed (finished with errors)
        """

        # get the path to the taskstatus log file
        task_log = torun_dict['taskstatus_log']

        # read the LAST line of the log file
        with open(task_log, 'r') as f:
            lines = f.readlines()
            last_line = lines[-1]   
        
        # get the status from the START of the last line
        if last_line.startswith("finished successfully"):
            # delete the taskstatus log file
            os.remove(task_log)
            return 1
        elif last_line.startswith("failed") or last_line.startswith("  Error"):
            # don't delte log, might want to check!
            return 2
        elif last_line.startswith("running"):
            return 0
        # otherwise raise error
        else:
            raise ValueError("Taskstatus log file is corrupted or not in the correct format")


## ALL TASKS DEFINED BELOW

class TestTask1(Task):
    """ Useless task for testing purposes.
    """

    def __init__(self):
        super().__init__()
        self.name = "test1"
        self.prerequisites = []

    def _run(self, torun_dict, log) -> bool:
        # RUN TASK!!!
        print(f"    Running {self.name} task on {os.path.join(torun_dict['fly_dir'], torun_dict['trial'])}")
        time.sleep(2)
        print(f"    {self.name} task done")

class TestTask2(Task):
    """ Useless task 2 for testing purposes.
    """

    def __init__(self):
        super().__init__()
        self.name = "test2"
        self.prerequisites = ['test1']

    def _run(self, torun_dict, log) -> bool:
        # RUN TASK!!!
        print(f"    Running {self.name} task on {os.path.join(torun_dict['fly_dir'], torun_dict['trial'])}")
        time.sleep(2)
        print(f"    {self.name} task 2 done")
    
class TifTask(Task):
    """
    Task to convert .raw files to .tif files.
    Will save the .tif files in the same directory as the .raw files,
    named as stack.tif (1 channel) or stack_ch1.tif + stack_ch2.tif (two channels)
    """
    def __init__(self, prio=0):
        self.name = "tif"
        self.prerequisites = []

    def _run(self, torun_dict, log):
        # convert raw to tiff
        create_tiffs(self.full_path(torun_dict))

class FictracTask(Task):
    """ 
    Task to run fictrac to track the ball movement and save the results in the behaviour dataframe.
    """

    def __init__(self):
        super().__init__()
        self.name = "fictrac"
        self.prerequisites = []

    def _run(self, torun_dict, log) -> bool:
        # check if overwrite = True
        # if so, show a warning (fictrac will make a new file, not really overwrite)
        if torun_dict['overwrite']:
            log.add_line_to_log("WARNING: Overwrite = True for fictrac task will be ignored, as fictrac will create a new file regardless.")

        # run fictrac
        config_and_run_fictrac(self.full_path(torun_dict))
        # convert output to df and save it
        fictract_df_path = get_fictrac_df(self.full_path(torun_dict))
        print('fine')
        # combine the fictrac df with the main df
        print(self.full_path(torun_dict))
        print(fictract_df_path)
        combine_df(self.full_path(torun_dict), fictract_df_path, log)





# # TEMPLATE FOR NEW TASKS
# class TaskName(Task):
#     """ Enter task description here.
#     """

#     def __init__(self):
#         super().__init__()
#         self.name = "name"
#         self.prerequisites = ['prerequisite_1_taskname', 'prerequisite_2_taskname', ...]

#     def _run(self, torun_dict, log) -> bool:
#         # enter functions to run here
#         # DO NOT write specific code lines here, use EXTERNAL FUNCTIONS instead
        


## END OF TASK DEFINITIONS



## Create the task_collection dictionary automatically
# dict format: {task_name: TaskClass}
task_collection = {cls().name: cls for cls in Task.__subclasses__()}
