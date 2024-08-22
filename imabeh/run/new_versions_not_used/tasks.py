#task_collection

"""
sub-module to define different steps of pre-processing as sub-classes of Task.
List of all available tasks is defined in task_collection at the bottom - REMEMBER TO ADD NEW TASKS THERE!
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


from imabeh.imabeh.run.new_versions_not_used.runutils import add_line_to_log


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
        self.previous_tasks = []

    



## ALL TASKS DEFINED BELOW

class TestTask(Task):
    """ Useless task for testing purposes.
    
    Returns
    -------
    success : bool
        True if the task ran successfully, False otherwise.
    """

    def __init__(self):
        super().__init__()
        self.name = "test"
        self.previous_tasks = []

    def run(self, trial_path, log_path):

        # log the start of the task
        line = f"{time.ctime(time.time())}: starting {self.name} task for fly {trial_path}"
        add_line_to_log(log_path, line)

        try:
            # RUN TASK!!!
            print("Running test task")
            time.sleep(5)
            print("Test task done")

            success = True
        except:
            success = False

        return success



## List of all tasks available to run (defined above)
task_collection = {
    "test": TestTask,
}