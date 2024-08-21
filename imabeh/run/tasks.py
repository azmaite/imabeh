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

from imabeh.run.runutils import add_line_to_log


# # List of all tasks available to run (defined below)
# task_collection = {
#     "tif": TifTask,
#     "df": DfTask,
#     "fictrac": FictracTask,
#     "sleap": SleapTask,
#     "sleapRL": SleapRLTask,
#     "df3d": Df3dTask,
#     "laser_stim_process": LaserStimProcessTask,
# }

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

    



# ## ALL TASKS DEFINED BELOW

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



# class TifTask(Task):
#     """
#     Task to convert .raw files to .tif files
#     """
#     def __init__(self, prio=0):
#         super().__init__(prio)
#         self.name = "tif"
#         self.previous_tasks = [TwopDataTransferTask()]

#     def test_todo(self, fly_dict):
#         TODO1 = self._test_todo_trials(fly_dict, file_name=global_params.green_raw)
#         TODO2 = self._test_todo_trials(fly_dict, file_name=global_params.red_raw)
#         return TODO1 and TODO2  # only do this task if both are missing

#     def run(self, fly_dict, params=None):
#         if not self.wait_for_previous_task(fly_dict):
#             return False
#         else:
#             self.send_status_email(fly_dict)
#             print(f"{time.ctime(time.time())}: starting {self.name} task for fly {fly_dict['dir']}")

#         self.params = deepcopy(params) if params is not None else deepcopy(global_params)

#         trial_dirs = get_selected_trials(fly_dict)

#         self.params.twoway_align = False
#         self.params.ref_frame = ""  # -> don't save ref frame
#         self.params.overwrite = fly_dict["overwrite"]

#         preprocess = PreProcessFly(fly_dir=fly_dict["dir"], params=self.params,
#                                    trial_dirs=trial_dirs)
#         # convert raw to tiff
#         preprocess.run_all_trials()
#         return True

# class DfTask(Task):
#     """
#     Task to run create beh_df and twop_df without running fictrac or wheel processing.
#     """
#     def __init__(self, prio=0):
#         super().__init__(prio)
#         self.name = "df"
#         self.previous_tasks = [BehDataTransferTask(), SyncDataTransfer()]
#     def test_todo(self, fly_dict):
#         return self._test_todo_trials(fly_dict, file_name=global_params.df3d_df_out_dir)
#     def run(self, fly_dict, params=None):
#         if not self.wait_for_previous_task(fly_dict):
#             return False
#         else:
#             self.send_status_email(fly_dict)
#             print(f"{time.ctime(time.time())}: starting {self.name} task for fly {fly_dict['dir']}")

#         self.params = deepcopy(params) if params is not None else deepcopy(global_params)
#         self.params.overwrite = fly_dict["overwrite"]
#         self.params.ball_tracking = None
#         self.params.denoise_params.pre_post_frame = 0

#         trial_dirs = get_selected_trials(fly_dict)

#         print("STARTING PREPROCESSING OF FLY: \n" + fly_dict["dir"])
#         preprocess = PreProcessFly(fly_dir=fly_dict["dir"], params=self.params,
#                                    trial_dirs=trial_dirs)
#         preprocess.get_dfs()
#         return True

# class FictracTask(Task):
#     """
#     Task to run fictrac to track the ball movement and save the results in the behaviour dataframe
#     """
#     def __init__(self, prio=0):
#         super().__init__(prio)
#         self.name = "fictrac"
#         self.previous_tasks = [BehDataTransferTask(), SyncDataTransfer()]

#     def test_todo(self, fly_dict):
#         # print("TODO: implement FictracTask.test_todo() method!!!")
#         TODO1 = self._test_todo_trials(fly_dict, file_name=global_params.df3d_df_out_dir)
#         trial_dirs_todo = get_selected_trials(fly_dict)
#         found_files = [utils.find_file(os.path.join(trial_dir, "behData", "images"),
#                                        name=f"camera_{CURRENT_USER['fictrac_cam']}-*.dat",
#                                        raise_error=False) for trial_dir in trial_dirs_todo]
#         TODO2 = any([found_file is None for found_file in found_files])
#         if not TODO2:
#             TODO2 = any([bool(len(found_file)) is None for found_file in found_files])
#         return TODO1  or TODO2

#     def run(self, fly_dict, params=None):
#         if not self.wait_for_previous_task(fly_dict):
#             return False
#         else:
#             self.send_status_email(fly_dict)
#             print(f"{time.ctime(time.time())}: starting {self.name} task for fly {fly_dict['dir']}")

#         self.params = deepcopy(params) if params is not None else deepcopy(global_params)
#         self.params.overwrite = fly_dict["overwrite"]
#         # self.params.denoise_params.pre_post_frame = 0

#         trial_dirs = get_selected_trials(fly_dict)
#         # this has no inbuilt override protection -
#         # -> only protected by the test_todo() method of this Task
#         config_and_run_fictrac(fly_dict["dir"], trial_dirs)

#         print("STARTING PREPROCESSING OF FLY: \n" + fly_dict["dir"])
#         preprocess = PreProcessFly(fly_dir=fly_dict["dir"], params=self.params,
#                                    trial_dirs=trial_dirs)
#         preprocess.get_dfs()
#         return True

#     def run_multiple_flies(self, fly_dicts, params=None):
#         self.params = deepcopy(params) if params is not None else deepcopy(global_params)
#         if isinstance(self.params, list):
#             n_params = len(self.params)
#         else:
#             n_params = 1
#         all_selected_trials = []
#         for fly_dict in fly_dicts:
#             all_selected_trials += get_selected_trials(fly_dict)

#         config_and_run_fictrac("multiple flies", all_selected_trials)

#         for i_fly, fly_dict in enumerate(fly_dicts):
#             print("STARTING PREPROCESSING OF FLY: \n" + fly_dict["dir"])
#             params = self.params if n_params == 1 else self.params[i_fly]
#             trial_dirs = get_selected_trials(fly_dict)
#             preprocess = PreProcessFly(fly_dir=fly_dict["dir"], params=params,
#                                        trial_dirs=trial_dirs)
#             preprocess.get_dfs()

# class SleapTask(Task):
#     """
#     Task to run simple and fast 2D pose estimation using Sleap 
#     and save results in behaviour dataframe.
#     Please install sleap according to instructions and create a conda environment called 'sleap' to use the capabilities of this module.
#     https://github.com/talmolab/sleap:
#     conda create -y -n sleap -c conda-forge -c nvidia -c sleap -c anaconda sleap
#     in case this does not work, try installing from source:
#     https://sleap.ai/installation.html#conda-from-source 

#     TODO: it is necessary to copy all files related to a trained sleap model into the following subfolder in order to use sleap:
#     twoppp/behaviour/sleap_model
#     an example model can be found here: 
#     /mnt/labserver/Ramdya-Lab/BRAUN_Jonas/Other/sleap/models/230516_135509.multi_instance.n=400
#     """
#     def __init__(self, prio: int=0) -> None:
#         super().__init__(prio)
#         self.name = "sleap"
#         self.previous_tasks = ["OR", FictracTask(), WheelTask(), DfTask()]

#     def test_todo(self, fly_dict: dict) -> bool:
#         trial_dirs_todo = get_selected_trials(fly_dict)
#         found_files = [utils.find_file(os.path.join(trial_dir, "behData", "images"),
#                                     name="sleap_output.h5",
#                                     raise_error=False) for trial_dir in trial_dirs_todo]
#         TODO = any([found_file is None for found_file in found_files])
#         return TODO

#     def run(self, fly_dict: dict, params: PreProcessParams=None) -> bool:
#         if not self.wait_for_previous_task(fly_dict):
#             return False
#         else:
#             self.send_status_email(fly_dict)
#             print(f"{time.ctime(time.time())}: starting {self.name} task for fly {fly_dict['dir']}")

#         self.params = deepcopy(params) if params is not None else deepcopy(global_params)
#         self.params.overwrite = fly_dict["overwrite"]

#         trial_dirs = get_selected_trials(fly_dict)

#         prepare_sleap(trial_dirs)
#         camera_num = "5" 
#         run_sleap(camera_num)
#         for trial_dir in trial_dirs:
#             beh_df = os.path.join(trial_dir, load.PROCESSED_FOLDER, global_params.df3d_df_out_dir)
#             beh_df = add_sleap_to_beh_df(trial_dir=trial_dir, beh_df=beh_df, out_dir=beh_df)
#         return True

# class SleapRLTask(Task):
#     """
#     Task to run simple and fast 2D pose estimation using Sleap 
#     and save results in behaviour dataframe.
    
#     This version (SleapRL) will run sleap for right side of fly (camera_5)
#     then on left side of fly (camera_1).
    
#     Please install sleap according to instructions and create a conda environment called 'sleap' to use the capabilities of this module.
#     https://github.com/talmolab/sleap:
#     conda create -y -n sleap -c conda-forge -c nvidia -c sleap -c anaconda sleap
#     in case this does not work, try installing from source:
#     https://sleap.ai/installation.html#conda-from-source 

#     TODO: it is necessary to add the sleap model address to the twopp/behavior/run_sleap_multiple_folders.sh script.
#     """
#     def __init__(self, prio: int=0) -> None:
#         super().__init__(prio)
#         self.name = "sleapRL"
#         self.previous_tasks = ["OR", FictracTask(), WheelTask(), DfTask()]

#     def test_todo(self, fly_dict: dict) -> bool:
#         trial_dirs_todo = get_selected_trials(fly_dict)
#         found_files = [utils.find_file(os.path.join(trial_dir, "behData", "images"),
#                                     name="sleap_output.h5",
#                                     raise_error=False) for trial_dir in trial_dirs_todo]
#         TODO = any([found_file is None for found_file in found_files])
#         return TODO

#     def run(self, fly_dict: dict, params: PreProcessParams=None) -> bool:
#         if not self.wait_for_previous_task(fly_dict):
#             return False
#         else:
#             self.send_status_email(fly_dict)
#             print(f"{time.ctime(time.time())}: starting {self.name} task for fly {fly_dict['dir']}")

#         self.params = deepcopy(params) if params is not None else deepcopy(global_params)
#         self.params.overwrite = fly_dict["overwrite"]

#         trial_dirs = get_selected_trials(fly_dict)
        
#         # delete sleap_output.h5 and L/R versions if present
#         for trial_dir in trial_dirs:
#             sleap_output = os.path.join(trial_dir, "behData", "images", "sleap_output.h5")
#             sleap_output_R = os.path.join(trial_dir, "behData", "images", "sleap_output_R.h5")
#             sleap_output_L = os.path.join(trial_dir, "behData", "images", "sleap_output_L.h5") 
#             if os.path.isfile(sleap_output):
#                 os.remove(sleap_output)
#             if os.path.isfile(sleap_output_R):
#                 os.remove(sleap_output_R)
#             if os.path.isfile(sleap_output_L):
#                 os.remove(sleap_output_L)
	
# 	# prepare sleap
#         prepare_sleap(trial_dirs)
        
#         # run sleap on right side
#         camera_num = "5" 
#         run_sleap(camera_num)        
#         # rename body parts in sleap_output.h5 to add 'R_'
#         for trial_dir in trial_dirs:
#             sleap_output = os.path.join(trial_dir, "behData", "images", "sleap_output.h5")
#             with h5py.File(sleap_output, 'r+') as file:
#                 nodes = file['node_names'][:]
#                 for i, node in enumerate(nodes):
#                     node_str = node.decode('utf-8')
#                     node_str = 'R_' + node_str
#                     nodes[i] = node_str.encode('utf-8')
#                 file['node_names'][:] = nodes
                    
#         # transfer to df file
#         for trial_dir in trial_dirs:
#             beh_df = os.path.join(trial_dir, load.PROCESSED_FOLDER, global_params.df3d_df_out_dir)
#             beh_df = add_sleap_to_beh_df(trial_dir=trial_dir, beh_df=beh_df, out_dir=beh_df)
            
#         # rename sleap output file for R side  
#         for trial_dir in trial_dirs:  
#             sleap_output = os.path.join(trial_dir, "behData", "images", "sleap_output.h5")
#             sleap_output_R = os.path.join(trial_dir, "behData", "images", "sleap_output_R.h5")
#             os.rename(sleap_output, sleap_output_R)
        
#         for trial_dir in trial_dirs:
#             # run sleap on left side
#             camera_num = "1" 
#             run_sleap(camera_num)
#             # rename body parts in sleap_output.h5 to add 'L_'
#             with h5py.File(sleap_output, 'r+') as file:
#                 nodes = file['node_names'][:]
#                 for i, node in enumerate(nodes):
#                     node_str = node.decode('utf-8')
#                     node_str = 'L_' + node_str
#                     nodes[i] = node_str.encode('utf-8')
#                 file['node_names'][:] = nodes
                
#         # transfer to df file
#         for trial_dir in trial_dirs:
#             beh_df = os.path.join(trial_dir, load.PROCESSED_FOLDER, global_params.df3d_df_out_dir)
#             print(beh_df)
#             beh_df = add_sleap_to_beh_df(trial_dir=trial_dir, beh_df=beh_df, out_dir=beh_df)
        
#         # rename sleap output file for L side   
#         for trial_dir in trial_dirs: 
#             sleap_output = os.path.join(trial_dir, "behData", "images", "sleap_output.h5")
#             sleap_output_L = os.path.join(trial_dir, "behData", "images", "sleap_output_L.h5")
#             os.rename(sleap_output, sleap_output_L)
        
#         return True
        
# class Df3dTask(Task):
#     """
#     Task to run pose estimation using DeepFly3D and DF3D post processing
#     and save results in behaviour dataframe.
#     """
#     def __init__(self, prio: int=0) -> None:
#         super().__init__(prio)
#         self.name = "df3d"
#         self.previous_tasks = [BehDataTransferTask(), SyncDataTransfer()]

#     def test_todo(self, fly_dict: dict) -> bool:
#         TODO1 = self._test_todo_trials(fly_dict, file_name=global_params.df3d_df_out_dir)
#         trial_dirs_todo = get_selected_trials(fly_dict)
#         TODO2 = not all([os.path.isdir(os.path.join(trial_dir, "behData", "images", "df3d"))
#                          for trial_dir in trial_dirs_todo])
#         if not TODO2:
#             found_files = [utils.find_file(os.path.join(trial_dir, "behData", "images", "df3d"),
#                                        name="aligned_pose__*.pkl",
#                                        raise_error=False) for trial_dir in trial_dirs_todo]
#             TODO2 = any([found_file is None for found_file in found_files])
#             if not TODO2:
#                 TODO2 = not all([bool(len(found_file)) is None for found_file in found_files])

#         return TODO1 or TODO2

#     def run(self, fly_dict: dict, params: PreProcessParams=None) -> bool:
#         if not self.wait_for_previous_task(fly_dict):
#             return False
#         else:
#             self.send_status_email(fly_dict)
#             print(f"{time.ctime(time.time())}: starting {self.name} task for fly {fly_dict['dir']}")

#         self.params = deepcopy(params) if params is not None else deepcopy(global_params)
#         self.params.overwrite = fly_dict["overwrite"]

#         trial_dirs = get_selected_trials(fly_dict)

#         print("STARTING PREPROCESSING OF FLY: \n" + fly_dict["dir"])
#         preprocess = PreProcessFly(fly_dir=fly_dict["dir"], params=self.params,
#                                    trial_dirs=trial_dirs)
#         preprocess._pose_estimate()
#         preprocess._post_process_pose()
#         return True

#         if not self.wait_for_previous_task(fly_dict):
#             return False
#         else:
#             self.send_status_email(fly_dict)
#             print(f"{time.ctime(time.time())}: starting {self.name} task for fly {fly_dict['dir']}")

#         self.params = deepcopy(params) if params is not None else deepcopy(global_params)

#         trial_dirs = get_selected_trials(fly_dict)

#         self.params.dff_video_max_length = None # 100
#         self.params.dff_video_downsample = 2
#         self.params.overwrite = self.params.overwrite = fly_dict["overwrite"]

#         preprocess = PreProcessFly(fly_dir=fly_dict["dir"], params=self.params,
#                                    trial_dirs=trial_dirs)
#         for i, trial_dir in enumerate(trial_dirs):
#             preprocess._make_dff_behaviour_video_trial(i_trial=i, mask=None, include_2p=True)
#             shutil.copy2(
#                 os.path.join(trial_dir, load.PROCESSED_FOLDER,
#                 f"{preprocess.date}_{preprocess.genotype}_Fly{preprocess.fly}_" +\
#                 f"{preprocess.trial_names[i]}_{preprocess.params.dff_beh_video_name}.mp4"),
#                 CURRENT_USER["video_dir"])

#         return True

# class LaserStimProcessTask(Task):
#     """
#     Task to process the synchronisation signals from laser stimulation
#     """
#     def __init__(self, prio=0):
#         super().__init__(prio)
#         self.name = "laser_stim_process"
#         self.previous_tasks = ["OR", FictracTask(), WheelTask(), DfTask()]

#     def test_todo(self, fly_dict):
#         TODO1 = self._test_todo_trials(fly_dict, file_name="stim_paradigm.pkl")
#         TODO2 = self._test_todo_trials(fly_dict, file_name=global_params.df3d_df_out_dir)
#         return TODO1 or TODO2

#     def run(self, fly_dict, params=None):
#         if not self.wait_for_previous_task(fly_dict):
#             return False
#         else:
#             self.send_status_email(fly_dict)
#             print(f"{time.ctime(time.time())}: starting {self.name} task for fly {fly_dict['dir']}")

#         self.params = deepcopy(params) if params is not None else deepcopy(global_params)
#         self.params.overwrite = fly_dict["overwrite"]

#         trial_dirs = get_selected_trials(fly_dict)

#         for trial_dir in trial_dirs:
#             beh_df = os.path.join(trial_dir, load.PROCESSED_FOLDER, self.params.df3d_df_out_dir)
            
#             _ = get_sync_signals_stimulation(trial_dir,
#                                              sync_out_file="stim_sync.pkl",
#                                              paradigm_out_file="stim_paradigm.pkl",
#                                              overwrite=self.params.overwrite,
#                                              index_df=beh_df,
#                                              df_out_dir=beh_df)
            
#             twop_df = os.path.join(trial_dir, load.PROCESSED_FOLDER, self.params.twop_df_out_dir)
#             # change self.previous_task temporarily to only persue behavioural data processing
#             # if there actually is behavioural data
#             _ = get_beh_info_to_twop_df(beh_df, twop_df, twop_df_out_dir=twop_df)
#             previous_tasks = deepcopy(self.previous_tasks)
#             self.previous_tasks = ["OR", FictracTask(), WheelTask()]
#             if os.path.isfile(twop_df) and self.test_previous_task_ready(fly_dict):
#                 _ = add_beh_state_to_twop_df(twop_df, twop_df_out_dir=twop_df)
#             self.previous_tasks = previous_tasks
#         return True


# List of all tasks available to run (defined above)
task_collection = {
    "test": TestTask,
}