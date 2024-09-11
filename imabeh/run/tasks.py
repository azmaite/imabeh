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
from imabeh.imaging2p import utils2p
from imabeh.behavior import fictrac, df3d
from imabeh.general import main


# general task class
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
        # this will keep detailed information about the task in case of failure, including errors
        # it can also be used to check if the task has finished for non python/bash tasks
        task_log = LogManager(log_name = f"_task_{self.name}_status")
        task_log.add_line_to_log("running started at " + datetime.now().isoformat(sep=' '))

        try:
            # RUN TASK!!!
            # for tasks run in python/bash, when the script runs to the end, it will return finished = True
            # for taks run elsewhere (ex. the cluster), it will return finished = False. In this case,
            # the taskstatus log file will be used to check if the task has finished
            # (make sure to implement the test_finished method in the Task subclass for these cases!!!!)
            finished = self._run(torun_dict, log)

            # log the correct end of the task - if finished
            if finished:
                task_log.add_line_to_log("finished successfully at " + time.ctime(time.time()))
            else:
                # return the path to the taskstatus log file
                return os.path.join(task_log.log_folder,task_log.log_file)

        except Exception as e:
            # log the failure of the task
            task_log.add_line_to_log("failed at " + time.ctime(time.time()))
            task_log.add_line_to_log(f"  Error: {e}")


    def _run(self, torun_dict):
        """
        abstract method to be re-used in each Task subclass to actually run the task
        """
        raise NotImplementedError("This method should be implemented in the subclass")

    def test_finished(self, torun_dict: dict) -> int:
        """
        abstract method to check if the task has finished by reading the taskstatus_log file
        Should not be implemented for tasks that are run in python/bash, only for tasks run elsewhere (ex. the cluster)
        - For tasks run in python/bash, python will know when the task has finished by the end of the script (no need to check)
        - For tasks run elsewhere, the taskstatus log file should be used to check to see if the task has finished
          (every certain time, the TaskManager will check if the task has finished using this method)
        """

        raise NotImplementedError(f"Task {self.name} does not have a test_finished method implemented")


## ALL TASKS DEFINED BELOW
## ----------------------------------------------- ##

# Test tasks

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
        return True

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
        return True

# Imaging tasks

class TifTask(Task):
    """
    Task to convert .raw files to .tif files.
    Will save the .tif files in the same directory as the .raw files,
    named as stack.tif (1 channel) or stack_ch1.tif + stack_ch2.tif (two channels)
    """
    def __init__(self, prio=0):
        self.name = "tif"
        self.prerequisites = []

    def _run(self, torun_dict, log) -> bool:
        # convert raw to tiff
        utils2p.create_tiffs(torun_dict['full_path'])
        return True


# Behavior tasks

class DfTask(Task):
    """ 
    Task create a general behavior dataframe with info from Thorsync.
    Df3d and fictrac dataframes will be combined with this dataframe.
    This also gets the info on optogenetic stimulation from the thorsync file.
    """
    def __init__(self):
        super().__init__()
        self.name = "df"
        self.prerequisites = []

    def _run(self, torun_dict, log) -> bool:
        main.get_sync_df(torun_dict['full_path'])
        return True

class FictracTask(Task):
    """ 
    Task to run fictrac to track the ball movement and save the results in the main processed dataframe.
    """
    def __init__(self):
        super().__init__()
        self.name = "fictrac"
        self.prerequisites = ["df"]

    def _run(self, torun_dict, log) -> bool:
        try:
            # run fictrac and convert output to df
            fictrac.config_and_run_fictrac(torun_dict['full_path'])
            fictract_df_path = fictrac.get_fictrac_df(torun_dict['full_path'])
            # combine the fictrac df with the main processed df
            combine_df(torun_dict['full_path'], fictract_df_path, log)
        except Exception as e:
            log.add_line_to_log(f"Error running fictrac: {e}")
            raise e
        
        return True

class Df3dTask(Task):
    """ 
    Task to run pose estimation using DeepFly3D and Df3d post processing
    and save results in behaviour dataframe.
    """
    def __init__(self):
        super().__init__()
        self.name = "df3d"
        self.prerequisites = ["df"]

    def _run(self, torun_dict, log) -> bool:
        trial_dir = torun_dict['full_path']
        try:
            # run df3d, postprocess and get df
            df3d.run_df3d(trial_dir, log)
            df3d.postprocess_df3d_trial(trial_dir)
            df3d_df_path = df3d.get_df3d_df(trial_dir)
            # combine the df3d df with the main processed df
            combine_df(trial_dir, df3d_df_path, log)

        except Exception as e:
            log.add_line_to_log(f"Error running df3d: {e}")
            raise e
    
        return True


# # TEMPLATE FOR NEW TASKS
# class NameTask(Task):
#     """ Enter task description here.
#     """

#     def __init__(self):
#         super().__init__()
#         self.name = "name"
#         self.prerequisites = ['prerequisite_1_taskname', 'prerequisite_2_taskname', ...]

#     def _run(self, torun_dict, log) -> bool:
#         # try:
#           # enter functions to run here
#           # DO NOT write specific code lines here, use EXTERNAL FUNCTIONS instead
#         # except Exception as e:
#           # log.add_line_to_log(f"Error running name: {e}")
#           # raise e
#         return True


## END OF TASK DEFINITIONS



## Create the task_collection dictionary automatically
# dict format: {task_name: TaskClass}
task_collection = {cls().name: cls for cls in Task.__subclasses__()}#
