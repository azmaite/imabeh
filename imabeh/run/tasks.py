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

pipeline_dict: 
    found at the bottom of script, it is a dictionary of standard pipelines to run
    (each pipeline is a list of tasks)
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
from imabeh.imaging2p import utils2p, static2p
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
        finished : bool
            whether the task has finished or not
        task_log_path : str
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
        task_log_path = os.path.join(task_log.log_folder,task_log.log_file)

        try:
            # RUN TASK!!!
            # for tasks run in python/bash, when the script runs to the end, it will return finished = True
            # for taks run elsewhere (ex. the cluster), it will return finished = False. In this case,
            # the taskstatus log file will be used to check if the task has finished
            # (make sure to implement the test_finished method in the Task subclass for these cases!!!!)
            finished = self._run(torun_dict, log)

            # if finished, return the status and delete the task log file (so return an empty path)
            if finished:
                os.remove(task_log_path)
                return True, ''
            
            # if kept running, log and return task_log_path
            else:
                task_log.add_line_to_log("task running outside of python/bash at " + time.ctime(time.time()))
                return False, task_log_path

        except Exception as e:
            # log the failure of the task
            task_log.add_line_to_log("failed at " + time.ctime(time.time()))
            task_log.add_line_to_log(f"  Error: {e}")
            raise RuntimeError(f"Error running {self.name} task: {e}")


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

class FlattenTask(Task):
    """ 
    Task to flatten a set of Z stacks into a single average, registered STD projection
    """

    def __init__(self):
        super().__init__()
        self.name = "flat"
        self.prerequisites = ['tif']

    def _run(self, torun_dict, log) -> bool:
        try:
            # find tif image for channel_1
            stack_path = main.find_file(torun_dict['full_path'], "channel_1.tif", "tif")
            # flatten and save
            static2p.flatten_stack_std(stack_path)

        except Exception as e:
          log.add_line_to_log(f"Error running name: {e}")
          raise e
    
        return True # if task is run outside of python/bash, return False AND IMPLEMENT test_finished METHOD!!!



# Behavior tasks

class DfTask(Task):
    """ 
    Task create a general behavior dataframe with info from Thorsync.
    Df3d and fictrac dataframes will be combined with this dataframe if present.
    This also gets the info on optogenetic stimulation from the thorsync file.
    """
    def __init__(self):
        super().__init__()
        self.name = "df"
        self.prerequisites = []

    def _run(self, torun_dict, log) -> bool:
        # check if the main dataframe is already present, and create if not
        main_df_path = os.path.join(torun_dict['full_path'], user_config["processed_path"], "processed_df.pkl")
        if not os.path.exists(main_df_path):
            main.get_sync_df(torun_dict['full_path'])

        # add fictrac dataframe if present
        try:
            fictrac_dir = os.path.join(torun_dict['full_path'], user_config['fictrac_path'])
            fictrac_df_path = main.find_file(fictrac_dir, "fictrac_df.pkl", "fictrac df")
            combine_df(torun_dict['full_path'], fictrac_df_path, log)
        except FileNotFoundError:
            log.add_line_to_log("No fictrac dataframe found")
        except Exception as e:
            log.add_line_to_log(f"Error combining fictrac dataframe: {e}")

        # add df3d dataframe if present
        try:
            df3d_dir = os.path.join(torun_dict['full_path'], user_config['df3d_path'])
            df3d_df_path = main.find_file(df3d_dir, "df3d_df.pkl", "df3d df")
            combine_df(torun_dict['full_path'], df3d_df_path, log)
        except FileNotFoundError:
            log.add_line_to_log("No df3d dataframe found")
        except Exception as e:
            log.add_line_to_log(f"Error combining df3d dataframe: {e}")

        return True

class FictracTask(Task):
    """ 
    Task to run fictrac to track the ball movement and convert output to df.
    """
    def __init__(self):
        super().__init__()
        self.name = "fictrac"
        self.prerequisites = []

    def _run(self, torun_dict, log) -> bool:
        try:
            # run fictrac and convert output to df
            fictrac.config_and_run_fictrac(torun_dict['full_path'])
            _ = fictrac.get_fictrac_df(torun_dict['full_path'])
        except Exception as e:
            log.add_line_to_log(f"Error running fictrac: {e}")
            raise e
        
        return True

class Df3dTask(Task):
    """ 
    Task to run pose estimation using DeepFly3D and Df3d post processing
    and convert output to dataframe.
    """
    def __init__(self):
        super().__init__()
        self.name = "df3d"
        self.prerequisites = []

    def _run(self, torun_dict, log) -> bool:
        trial_dir = torun_dict['full_path']
        try:
            # run df3d, postprocess and get df
            df3d.run_df3d(trial_dir)
            df3d.postprocess_df3d_trial(trial_dir)
            _ = df3d.get_df3d_df(trial_dir)

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
#         try:
#           # enter functions to run here
#           # DO NOT write specific code lines here, use EXTERNAL FUNCTIONS instead
#         except Exception as e:
#           log.add_line_to_log(f"Error running name: {e}")
#           raise e
#         return True # if task is run outside of python/bash, return False AND IMPLEMENT test_finished METHOD!!!


## END OF TASK DEFINITIONS

## PIPELINES
# add new pipeline sets here as list of task names
pipeline_dict = {
    "test" : ["tif", "df"],
    "ablation_beh" : ["fictrac", "df3d", "df"],
    "ablation_stack" : ["tif", "flat"],
    }


## Create the task_collection dictionary automatically
# dict format: {task_name: TaskClass}
task_collection = {cls().name: cls for cls in Task.__subclasses__()}
