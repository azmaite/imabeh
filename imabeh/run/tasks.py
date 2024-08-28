#task_collection

"""
sub-module to define different steps of pre-processing as sub-classes of Task.
List of all available tasks is defined in task_collection at the bottom (automatically generated).
"""
import os
import time
# import shutil
# from copy import deepcopy
# import datetime
# from typing import List
# import numpy as np
# import h5py

# from twoppp import load, utils, TWOPPP_PATH
# from twoppp.register import warping
# from twoppp.pipeline import PreProcessFly, PreProcessParams
# from twoppp.behaviour.fictrac import config_and_run_fictrac
# from twoppp.behaviour.stimulation import get_sync_signals_stimulation, get_beh_info_to_twop_df, add_beh_state_to_twop_df
# from twoppp.behaviour.olfaction import get_sync_signals_olfaction
# from twoppp.behaviour.sleap import prepare_sleap, run_sleap, add_sleap_to_beh_df
# from twoppp.rois import prepare_roi_selection
# from twoppp.plot import show3d
# from twoppp.run.runparams import global_params, CURRENT_USER
# from twoppp.run.runutils import get_selected_trials, get_scratch_fly_dict, find_trials_2plinux
# from twoppp.run.runutils import send_email,split_fly_dict_trials

from imabeh.run.userpaths import LOCAL_DIR, user_config # get the current user configuration (paths and settings)
from imabeh.run.logmanager import LogManager


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
        log.add_line_to_log(f"{time.ctime(time.time())}: starting {self.name} task for trial {trial_path}")

        # log that the task is running into the taskstatus log file (create a new log)
        task_log = LogManager(log_name = f"_task_{self.name}_status")
        task_log.add_line_to_log("running started at " + time.ctime(time.time()))

        try:
            # RUN TASK!!!
            self._run(trial_path)

            # log the correct end of the task
            task_log.add_line_to_log("finished successfully at " + time.ctime(time.time()))

        except:
            # log the failure of the task
            task_log.add_line_to_log("failed at " + time.ctime(time.time()))

        # return the path to the taskstatus log file
        return os.path.join(task_log.log_folder,task_log.log_file)


    def _run(self, trial_path):
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
        elif last_line.startswith("failed"):
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

    def _run(self, trial_path) -> bool:
        # RUN TASK!!!
        print(f"    Running {self.name} task on {trial_path}")
        time.sleep(2)
        print(f"    {self.name} task done")


class TestTask2(Task):
    """ Useless task 2 for testing purposes.
    """

    def __init__(self):
        super().__init__()
        self.name = "test2"
        self.prerequisites = ['test1']

    def _run(self, trial_path) -> bool:
        # RUN TASK!!!
        print(f"    Running {self.name} task on {trial_path}")
        time.sleep(2)
        print(f"    {self.name} task 2 done")
    

# # TEMPLATE FOR NEW TASKS
# class TaskName(Task):
#     """ Enter task description here.
#     """

#     def __init__(self):
#         super().__init__()
#         self.name = "name"
#         self.prerequisites = ['prerequisite_1_taskname', 'prerequisite_2_taskname', ...]

#     def run(self, trial_path) -> bool:
#         # enter functions to run here
#         # DO NOT write specific code lines here, use EXTERNAL FUNCTIONS instead
        


## END OF TASK DEFINITIONS



## Create the task_collection dictionary automatically
# dict format: {task_name: TaskClass}
task_collection = {cls().name: cls for cls in Task.__subclasses__()}
