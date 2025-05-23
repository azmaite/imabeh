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
from datetime import datetime

# imports for the task manager to run
from imabeh.run.userpaths import LOCAL_DIR, user_config # get the current user configuration (paths and settings)
from imabeh.run.logmanager import LogManager

# task specific imports
from imabeh.imaging2p import utils2p, static2p
from imabeh.behavior import behavior, fictrac, df3d, sleap, videos
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
        log.add_line_to_log(f"{trial_path} - {self.name} start @ {datetime.now().isoformat(sep=' ')}, ")


        try:
            # RUN TASK!!!
            # for tasks run in python/bash, when the script runs to the end, it will return finished = True
            # for taks run elsewhere (ex. the cluster), it will return finished = False. In this case,
            # make sure to implement the test_finished method in the Task subclass!!!!
            finished = self._run(torun_dict, log)

            # if finished, return the status and update the general log
            if finished:
                log.add_line_to_log(f"end @ {datetime.now().isoformat(sep=' ')} \n")
                return True
            
            # if still running, log and return task_log_path
            else:
                log.add_line_to_log(f"running... \n")
                return False

        except Exception as e:
            # log the failure of the task in general log
            log.add_line_to_log(f"FAILED @ {datetime.now().isoformat(sep=' ')} \n")
            log.add_line_to_log(f"        Error: {e} \n")
            log.add_line_to_log(f"\n")
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
        try:
            # convert raw to tiff
            utils2p.create_tiffs(torun_dict['full_path'])

        except Exception as e:
          raise e
        
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
          raise e
    
        return True



# Behavior tasks

class DfTask(Task):
    """ 
    Task create a general behavior dataframe with info from Thorsync.
    Df3d, fictrac, and sleap dataframes will be combined with this dataframe if present.
    This also gets the info on optogenetic stimulation from the thorsync file.
    """
    def __init__(self):
        super().__init__()
        self.name = "df"
        self.prerequisites = []

    def _run(self, torun_dict, log) -> bool:
        try:
            # check if the main dataframe is already present
            # if yes, delete and create. If not, create it
            main_df_path = os.path.join(torun_dict['full_path'], user_config["processed_path"], "processed_df.pkl")
            if os.path.exists(main_df_path):
                os.remove(main_df_path)
            main.get_sync_df(torun_dict['full_path'])

            trial_dir = torun_dict['full_path']

            # add fictrac dataframe to main dataframe
            fictrac_dir = os.path.join(trial_dir, user_config['fictrac_path'])
            try:
                fictrac_df_path = main.find_file(fictrac_dir, "fictrac_df.pkl", "fictrac df")
                main.combine_df(trial_dir, fictrac_df_path, log)
            except FileNotFoundError:
                pass

            # add df3d dataframe to main df
            df3d_dir = os.path.join(trial_dir, user_config['df3d_path'])
            try:
                df3d_df_path = main.find_file(df3d_dir, "df3d_df.pkl", "df3d df")
                main.combine_df(trial_dir, df3d_df_path, log)
            except FileNotFoundError:
                pass


        
        except Exception as e:
            raise e
        
        return True

class FictracTask(Task):
    """ 
    Task to run fictrac to track the ball movement and convert output to df.
    Combine the resulting dataframe to the main behavior dataframe.
    """
    def __init__(self):
        super().__init__()
        self.name = "fictrac"
        self.prerequisites = ['df']

    def _run(self, torun_dict, log) -> bool:
        trial_dir = torun_dict['full_path']
        try:
            # run fictrac and convert output to df
            fictrac.config_and_run_fictrac(trial_dir)
            _ = fictrac.get_fictrac_df(trial_dir)

            # add fictrac dataframe to main dataframe
            fictrac_dir = os.path.join(trial_dir, user_config['fictrac_path'])
            fictrac_df_path = main.find_file(fictrac_dir, "fictrac_df.pkl", "fictrac df")
            main.combine_df(trial_dir, fictrac_df_path, log)

        except Exception as e:
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
        self.prerequisites = ['df']

    def _run(self, torun_dict, log) -> bool:
        trial_dir = torun_dict['full_path']
        try:
            # run df3d, postprocess and get df
            df3d.run_df3d(trial_dir)
            df3d.postprocess_df3d_trial(trial_dir)
            _ = df3d.get_df3d_df(trial_dir)

            # add df3d dataframe to main df
            df3d_dir = os.path.join(trial_dir, user_config['df3d_path'])
            df3d_df_path = main.find_file(df3d_dir, "df3d_df.pkl", "df3d df")
            main.combine_df(trial_dir, df3d_df_path, log)

            # make df3d videos on a subset of frames (aiming to get stimulation period)
            print('making video...')
            df3d.df3d_video(trial_dir, start_frame=700, end_frame=1100) 

        except Exception as e:
            raise e
    
        return True
   
class SleapTask(Task):
    """ Enter task description here.
    """

    def __init__(self):
        super().__init__()
        self.name = "sleap"
        self.prerequisites = ['df']

    def _run(self, torun_dict, log) -> bool:
        trial_dir = torun_dict['full_path']
        try:
            # run sleap and convert output to df
            sleap.run_sleap(trial_dir, camera_num='5')
            sleap.make_sleap_df(trial_dir)

            # add sleap dataframe to main df
            sleap_dir = os.path.join(trial_dir, 'behData/sleap')
            sleap_df_path = sleap_dir + "/sleap_df.pkl"
            main.combine_df(trial_dir, sleap_df_path, log)

        except Exception as e:
            raise e
        
        return True
    

class GridTask(Task):
    """ Creates a 3x3 video grid of all stimulations.
    """

    def __init__(self):
        super().__init__()
        self.name = "grid"
        self.prerequisites = ['df']

    def _run(self, torun_dict, log) -> bool:
        try:
            camera_num = 5
            trial_dir = torun_dict['full_path']
            videos.make_video_grid(trial_dir, camera_num)

        except Exception as e:
            raise e
        return True 
    

class SplitTask(Task):
    """ Splits video into subsections based on the stimulation times. Splits the df too.
    """

    def __init__(self):
        super().__init__()
        self.name = "split"
        self.prerequisites = ['df']

    def _run(self, torun_dict, log) -> bool:
        try:
            trial_dir = torun_dict['full_path']
            behavior.split_videos(trial_dir)

        except Exception as e:
            raise e
        return True


class SplitBehTask(Task):
    """ Runs df3d and fictrac on the split videos.
    """

    def __init__(self):
        super().__init__()
        self.name = "splitBeh"
        self.prerequisites = ['df', 'split']

    def _run(self, torun_dict, log) -> bool:
        try:
            # get list of split videos
            trial_images_dir = os.path.join(torun_dict['full_path'], 'behData', 'images')
            split_list = os.listdir(trial_images_dir)
            split_list = [i for i in split_list if 'stim_' in i]
            split_list.sort()

            for split in split_list:
                split_dir = os.path.join(trial_images_dir, split)

                try:

                    # run df3d on each split video
                    df3d.run_df3d(split_dir)
                    print('df3d done, postprocessing')
                    df3d.postprocess_df3d_trial(split_dir)
                    print('postprocessing done, getting df')
                    df3d.get_df3d_df(split_dir)

                    # make df3d videos on full duration of video
                    print('making video...')
                    #df3d.df3d_video(split_dir) 

                    # add df3d dataframe to main df
                    df3d_dir = os.path.join(split_dir, user_config['df3d_path'])
                    df3d_df_path = main.find_file(df3d_dir, "df3d_df.pkl", "df3d df")
                    main.combine_df(split_dir, df3d_df_path, log)
                
                except Exception as e:
                    log.add_line_to_log(f"  Failed df3d for {split}")

                # run fictrac on each split video (if the trial doesn't contain 'noball')
                if 'noball' not in split_dir:
                    fictrac.config_and_run_fictrac(split_dir)
                    _ = fictrac.get_fictrac_df(split_dir)

                    # add fictrac dataframe to main df
                    fictrac_dir = os.path.join(split_dir, user_config['fictrac_path'])
                    fictrac_df_path = main.find_file(fictrac_dir, "fictrac_df.pkl", "fictrac df")
                    main.combine_df(split_dir, fictrac_df_path, log)    

            # join all the split dataframes into one
            behavior.join_split_df(torun_dict['full_path'], log)


        except Exception as e:
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
#             # enter functions to run here
#             # DO NOT write specific code lines here, use EXTERNAL FUNCTIONS instead
#         except Exception as e:
#             raise e
#         return True # if task is run outside of python/bash, return False AND IMPLEMENT test_finished METHOD!!!


## END OF TASK DEFINITIONS

## PIPELINES
# add new task pipeline sets here as list of task names 
# OR full pipelines as lists of keywords + task names (ex. [["k-beh"],["grid", "df"]]) 
# this will replace both the trials and tasks
task_pipeline_dict = {
    "test" : ["tif", "df"],
}
full_pipeline_dict = {
    "ablation" : [[["k-beh"],["df3d", "fictrac", "df", "grid"]],
                  [["k-stack"],["tif", "flat"]]],
    }


## Create the task_collection dictionary automatically
# dict format: {task_name: TaskClass}
task_collection = {cls().name: cls for cls in Task.__subclasses__()}
