"""
sub-module to define different steps of pre-processing as sub-classes of Task.

Tasks can be run in python or other languages, e.g. bash scripts.
TO ADD A NEW TASK:
1. Define a new sub-class of Task
2. Add the task to the task collection dictionary at the end of this script
3. Make sure the task has a 'test_if_done' method to check whether the task has finished (with errors or not)
"""

import os
import time
import shutil
from copy import deepcopy
import datetime
from typing import List
import numpy as np
import h5py

from twoppp import load, utils, TWOPPP_PATH
from twoppp.register import warping
from twoppp.pipeline import PreProcessFly, PreProcessParams
from twoppp.behaviour.fictrac import config_and_run_fictrac
from twoppp.behaviour.stimulation import get_sync_signals_stimulation, get_beh_info_to_twop_df, add_beh_state_to_twop_df
from twoppp.behaviour.olfaction import get_sync_signals_olfaction
from twoppp.behaviour.sleap import prepare_sleap, run_sleap, add_sleap_to_beh_df
from twoppp.rois import prepare_roi_selection
from twoppp.plot import show3d
from twoppp.run.runparams import global_params, CURRENT_USER
from twoppp.run.runutils import get_selected_trials, get_scratch_fly_dict, find_trials_2plinux
from twoppp.run.runutils import send_email,split_fly_dict_trials

class Task:
    """
    Base class to implement a particular pre-processing step.
    """
    def __init__(self) -> None:
        """
        Base class to implement a particular pre-processing step.
        """
        # PROPERTIES
        self.name = ""
        self.params = None
        self.prerequisite_tasks = []


    def _test_if_done(self, trial_task_dict) -> int:
        """
        abstract method to be re-used in test_if_done() for each subtask
        Function to check whether the current task has finished.
        IMPLEMENT THIS IN EVERY SUBTASK!

        Parameters
        ----------
        trial_task_dict : dict
            dictionary containing the following information:
            - dir: str, the base directory of the fly, where the data is stored
            - trial: str, the trial directory
            - task: str, the task name
            - overwrite: bool, whether or not to force an overwrite of the previous results
            - args: list[str], additional arguments

        Returns
        -------
        done_status : int
            whether the task has been completed (1) or not (0) or gave an error (2)
        """
        done_status = 0

        return done_status


    def run(self, trial_task_dict: dict) -> bool:
        """
        run the current task on a particular fly.

        Parameters
        ----------
        trial_task_dict : dict
            dictionary containing the following information:
            - dir: str, the base directory of the fly, where the data is stored
            - trial: str, the trial directory
            - task: str, the task name
            - overwrite: bool, whether or not to force an overwrite of the previous results
            - args: list[str], additional arguments

        Returns
        -------
        bool
            success: True if finished, False if waiting for previous task
        """
        if not self.wait_for_previous_task(fly_dict):
            return False
        else:
            self.send_status_email(fly_dict)
            return True


    def test_previous_task_ready(self, fly_dict: dict) -> bool:
        """
        returns True if previous tasks are finished and False otherwise

        Parameters
        ----------
        fly_dict : dict
            dictionary containing the following information:
            - dir: str, the base directory of the fly, where the data is stored
            - selected_trials: a string describing which trials to run on,
                               e.g. "001,002" or "all_trials"
            - overwrite: bool, whether or not to force an overwrite of the previous results
            - status: bool, whether the todo is "ready","running", "done", or "waiting"
            - args: list[str], additional arguments

        Returns
        -------
        bool
            True if all previous tasks have finished, False otherwise

        Raises
        ------
        NotImplementedError
            if not isinstance(self.previous_tasks, list)
        """
        if isinstance(self.previous_tasks, list):
            if "OR" in self.previous_tasks:
                previous_tasks = deepcopy(self.previous_tasks)
                OR = previous_tasks.pop(previous_tasks.index("OR"))
                fly_dicts_split_trials = split_fly_dict_trials(fly_dict)
                trials_ready = []
                for trial_fly_dict in fly_dicts_split_trials:
                    trials_ready.append(np.logical_or.reduce([not task.test_todo(trial_fly_dict) for task in previous_tasks]))
                return all(trials_ready)
            else:
                return not any([task.test_todo(fly_dict) for task in self.previous_tasks])
        else:
            raise NotImplementedError
        

    def wait_for_previous_task(self, fly_dict: dict) -> bool:
        """
        returns True if all previous tasks have finished.
        If previous task is not finished, waits for self.t_wait_s
        and checks again whether previous task is finished
        if previous task is still not finished, return False. If finished, returns True

        Parameters
        ----------
        fly_dict : dict
            dictionary containing the following information:
            - dir: str, the base directory of the fly, where the data is stored
            - selected_trials: a string describing which trials to run on,
                               e.g. "001,002" or "all_trials"
            - overwrite: bool, whether or not to force an overwrite of the previous results
            - status: bool, whether the todo is "ready","running", "done", or "waiting"
            - args: list[str], additional arguments

        Returns
        -------
        bool
            True if all previous tasks have finished, False otherwise
        """
        if self.t_wait_s:
            if not self.test_previous_task_ready(fly_dict):
                names = [task.name for task in self.previous_tasks]
                print(f"Waiting for tasks {names} of fly {fly_dict['dir']} to finish.")
                time.sleep(self.t_wait_s)
                return self.test_previous_task_ready(fly_dict)
            return True
        else:
            return self.test_previous_task_ready(fly_dict)


## DEFINE TASKS HERE AS A SUBCLASS OF Task
# each task MUST have a test_finished() method to check whether the task has finished and if it gave an error
# most should also have a run() method to run the task

class TwopDataTransferTask(Task):
    """
    Dummy Task to check whether twop photon data was already transfered
    """
    def __init__(self):
        self.name = "twop_data_transfer"

    def test_todo(self, trial_task_dict):
        if CURRENT_USER["check_2plinux_trials"]:
            twoplinux_trial_names = find_trials_2plinux(fly_dict["dir"],
                                                        CURRENT_USER["initials"], twop=True)
        else:
            twoplinux_trial_names = None
        trial_dirs_todo = get_selected_trials(fly_dict, twoplinux_trial_names)

        for trial_dir in trial_dirs_todo:
            if utils.find_file(trial_dir, "Image_0001_0001.raw", raise_error=False) is None \
                and utils.find_file(trial_dir, "Image_001_001.raw", raise_error=False) is None:  # new Thorsync
                return True
            if utils.find_file(trial_dir, "Experiment.xml", raise_error=False) is None:
                return True
        return False

class SyncDataTransfer(Task):
    """
    Dummy Task to check whether ThorSync data was already transfered
    """
    def __init__(self, prio=0):
        super().__init__(prio)
        self.name = "sync_data_transfer"

    def test_todo(self, fly_dict):
        if CURRENT_USER["check_2plinux_trials"]:
            twoplinux_trial_names = find_trials_2plinux(fly_dict["dir"],
                                                        CURRENT_USER["initials"], twop=True)
        else:
            twoplinux_trial_names = None
        trial_dirs_todo = get_selected_trials(fly_dict, twoplinux_trial_names)

        for trial_dir in trial_dirs_todo:
            if utils.find_file(trial_dir, "Episode001.h5", raise_error=False) is None:
                return True
            if utils.find_file(trial_dir, "ThorRealTimeDataSettings.xml",raise_error=False) is None:
                return True
        return False

class BehDataTransferTask(Task):
    """
    Dummy Task to check whether behavioural data was already transfered
    """
    def __init__(self, prio=0):
        super().__init__(prio)
        self.name = "beh_data_transfer"

    def test_todo(self, fly_dict):
        if CURRENT_USER["check_2plinux_trials"]:
            twoplinux_trial_names = find_trials_2plinux(fly_dict["dir"],
                                                        CURRENT_USER["initials"], twop=False)
        else:
            twoplinux_trial_names = None
        trial_dirs_todo = get_selected_trials(fly_dict, twoplinux_trial_names)

        for trial_dir in trial_dirs_todo:
            if utils.find_file(trial_dir, "capture_metadata.json", raise_error=False) is None:
                return True
            cam_name = f"camera_{CURRENT_USER['fictrac_cam']}.mp4"
            if utils.find_file(trial_dir, cam_name, raise_error=False) is None:
                return True
        return False

class TifTask(Task):
    """
    Task to convert .raw files to .tif files
    """
    def __init__(self, prio=0):
        super().__init__(prio)
        self.name = "tif"
        self.previous_tasks = [TwopDataTransferTask()]

    def test_todo(self, fly_dict):
        TODO1 = self._test_todo_trials(fly_dict, file_name=global_params.green_raw)
        TODO2 = self._test_todo_trials(fly_dict, file_name=global_params.red_raw)
        return TODO1 and TODO2  # only do this task if both are missing

    def run(self, fly_dict, params=None):
        if not self.wait_for_previous_task(fly_dict):
            return False
        else:
            self.send_status_email(fly_dict)
            print(f"{time.ctime(time.time())}: starting {self.name} task for fly {fly_dict['dir']}")

        self.params = deepcopy(params) if params is not None else deepcopy(global_params)

        trial_dirs = get_selected_trials(fly_dict)

        self.params.twoway_align = False
        self.params.ref_frame = ""  # -> don't save ref frame
        self.params.overwrite = fly_dict["overwrite"]

        preprocess = PreProcessFly(fly_dir=fly_dict["dir"], params=self.params,
                                   trial_dirs=trial_dirs)
        # convert raw to tiff
        preprocess.run_all_trials()
        return True


class FictracTask(Task):
    """
    Task to run fictrac to track the ball movement and save the results in the behaviour dataframe
    """
    def __init__(self, prio=0):
        super().__init__(prio)
        self.name = "fictrac"
        self.previous_tasks = [BehDataTransferTask(), SyncDataTransfer()]

    def test_todo(self, fly_dict):
        # print("TODO: implement FictracTask.test_todo() method!!!")
        TODO1 = self._test_todo_trials(fly_dict, file_name=global_params.df3d_df_out_dir)
        trial_dirs_todo = get_selected_trials(fly_dict)
        found_files = [utils.find_file(os.path.join(trial_dir, "behData", "images"),
                                       name=f"camera_{CURRENT_USER['fictrac_cam']}-*.dat",
                                       raise_error=False) for trial_dir in trial_dirs_todo]
        TODO2 = any([found_file is None for found_file in found_files])
        if not TODO2:
            TODO2 = any([bool(len(found_file)) is None for found_file in found_files])
        return TODO1  or TODO2

    def run(self, fly_dict, params=None):
        if not self.wait_for_previous_task(fly_dict):
            return False
        else:
            self.send_status_email(fly_dict)
            print(f"{time.ctime(time.time())}: starting {self.name} task for fly {fly_dict['dir']}")

        self.params = deepcopy(params) if params is not None else deepcopy(global_params)
        self.params.overwrite = fly_dict["overwrite"]
        # self.params.denoise_params.pre_post_frame = 0

        trial_dirs = get_selected_trials(fly_dict)
        # this has no inbuilt override protection -
        # -> only protected by the test_todo() method of this Task
        config_and_run_fictrac(fly_dict["dir"], trial_dirs)

        print("STARTING PREPROCESSING OF FLY: \n" + fly_dict["dir"])
        preprocess = PreProcessFly(fly_dir=fly_dict["dir"], params=self.params,
                                   trial_dirs=trial_dirs)
        preprocess.get_dfs()
        return True

    def run_multiple_flies(self, fly_dicts, params=None):
        self.params = deepcopy(params) if params is not None else deepcopy(global_params)
        if isinstance(self.params, list):
            n_params = len(self.params)
        else:
            n_params = 1
        all_selected_trials = []
        for fly_dict in fly_dicts:
            all_selected_trials += get_selected_trials(fly_dict)

        config_and_run_fictrac("multiple flies", all_selected_trials)

        for i_fly, fly_dict in enumerate(fly_dicts):
            print("STARTING PREPROCESSING OF FLY: \n" + fly_dict["dir"])
            params = self.params if n_params == 1 else self.params[i_fly]
            trial_dirs = get_selected_trials(fly_dict)
            preprocess = PreProcessFly(fly_dir=fly_dict["dir"], params=params,
                                       trial_dirs=trial_dirs)
            preprocess.get_dfs()

class WheelTask(Task):
    """
    Task to run wheel detection and velocity analysis & save the results in the behaviour dataframe
    """
    def __init__(self, prio=0):
        super().__init__(prio)
        self.name = "wheel"
        self.previous_tasks = [BehDataTransferTask(), SyncDataTransfer()]

    def test_todo(self, fly_dict):
        TODO1 = self._test_todo_trials(fly_dict, file_name=global_params.df3d_df_out_dir)
        trial_dirs_todo = get_selected_trials(fly_dict)
        found_files = [utils.find_file(os.path.join(trial_dir, "behData", "images"),
                                       name="wheel_locations.pkl",
                                       raise_error=False) for trial_dir in trial_dirs_todo]
        TODO2 = any([found_file is None for found_file in found_files])
        if not TODO2:
            TODO2 = any([bool(len(found_file)) is None for found_file in found_files])
        return TODO1  or TODO2

    def run(self, fly_dict, params=None):
        if not self.wait_for_previous_task(fly_dict):
            return False
        else:
            self.send_status_email(fly_dict)
            print(f"{time.ctime(time.time())}: starting {self.name} task for fly {fly_dict['dir']}")

        self.params = deepcopy(params) if params is not None else deepcopy(global_params)
        self.params.overwrite = fly_dict["overwrite"]
        self.params.ball_tracking = "wheel"

        trial_dirs = get_selected_trials(fly_dict)

        print("STARTING PREPROCESSING OF FLY: \n" + fly_dict["dir"])
        preprocess = PreProcessFly(fly_dir=fly_dict["dir"], params=self.params,
                                   trial_dirs=trial_dirs)
        preprocess.get_dfs()
        return True

class SleapTask(Task):
    """
    Task to run simple and fast 2D pose estimation using Sleap 
    and save results in behaviour dataframe.
    Please install sleap according to instructions and create a conda environment called 'sleap' to use the capabilities of this module.
    https://github.com/talmolab/sleap:
    conda create -y -n sleap -c conda-forge -c nvidia -c sleap -c anaconda sleap
    in case this does not work, try installing from source:
    https://sleap.ai/installation.html#conda-from-source 

    TODO: it is necessary to copy all files related to a trained sleap model into the following subfolder in order to use sleap:
    twoppp/behaviour/sleap_model
    an example model can be found here: 
    /mnt/labserver/Ramdya-Lab/BRAUN_Jonas/Other/sleap/models/230516_135509.multi_instance.n=400
    """
    def __init__(self, prio: int=0) -> None:
        super().__init__(prio)
        self.name = "sleap"
        self.previous_tasks = ["OR", FictracTask(), WheelTask(), DfTask()]

    def test_todo(self, fly_dict: dict) -> bool:
        trial_dirs_todo = get_selected_trials(fly_dict)
        found_files = [utils.find_file(os.path.join(trial_dir, "behData", "images"),
                                    name="sleap_output.h5",
                                    raise_error=False) for trial_dir in trial_dirs_todo]
        TODO = any([found_file is None for found_file in found_files])
        return TODO

    def run(self, fly_dict: dict, params: PreProcessParams=None) -> bool:
        if not self.wait_for_previous_task(fly_dict):
            return False
        else:
            self.send_status_email(fly_dict)
            print(f"{time.ctime(time.time())}: starting {self.name} task for fly {fly_dict['dir']}")

        self.params = deepcopy(params) if params is not None else deepcopy(global_params)
        self.params.overwrite = fly_dict["overwrite"]

        trial_dirs = get_selected_trials(fly_dict)

        prepare_sleap(trial_dirs)
        camera_num = "5" 
        run_sleap(camera_num)
        for trial_dir in trial_dirs:
            beh_df = os.path.join(trial_dir, load.PROCESSED_FOLDER, global_params.df3d_df_out_dir)
            beh_df = add_sleap_to_beh_df(trial_dir=trial_dir, beh_df=beh_df, out_dir=beh_df)
        return True

class SleapRLTask(Task):
    """
    Task to run simple and fast 2D pose estimation using Sleap 
    and save results in behaviour dataframe.
    
    This version (SleapRL) will run sleap for right side of fly (camera_5)
    then on left side of fly (camera_1).
    
    Please install sleap according to instructions and create a conda environment called 'sleap' to use the capabilities of this module.
    https://github.com/talmolab/sleap:
    conda create -y -n sleap -c conda-forge -c nvidia -c sleap -c anaconda sleap
    in case this does not work, try installing from source:
    https://sleap.ai/installation.html#conda-from-source 

    TODO: it is necessary to add the sleap model address to the twopp/behavior/run_sleap_multiple_folders.sh script.
    """
    def __init__(self, prio: int=0) -> None:
        super().__init__(prio)
        self.name = "sleapRL"
        self.previous_tasks = ["OR", FictracTask(), WheelTask(), DfTask()]

    def test_todo(self, fly_dict: dict) -> bool:
        trial_dirs_todo = get_selected_trials(fly_dict)
        found_files = [utils.find_file(os.path.join(trial_dir, "behData", "images"),
                                    name="sleap_output.h5",
                                    raise_error=False) for trial_dir in trial_dirs_todo]
        TODO = any([found_file is None for found_file in found_files])
        return TODO

    def run(self, fly_dict: dict, params: PreProcessParams=None) -> bool:
        if not self.wait_for_previous_task(fly_dict):
            return False
        else:
            self.send_status_email(fly_dict)
            print(f"{time.ctime(time.time())}: starting {self.name} task for fly {fly_dict['dir']}")

        self.params = deepcopy(params) if params is not None else deepcopy(global_params)
        self.params.overwrite = fly_dict["overwrite"]

        trial_dirs = get_selected_trials(fly_dict)
        
        # delete sleap_output.h5 and L/R versions if present
        for trial_dir in trial_dirs:
            sleap_output = os.path.join(trial_dir, "behData", "images", "sleap_output.h5")
            sleap_output_R = os.path.join(trial_dir, "behData", "images", "sleap_output_R.h5")
            sleap_output_L = os.path.join(trial_dir, "behData", "images", "sleap_output_L.h5") 
            if os.path.isfile(sleap_output):
                os.remove(sleap_output)
            if os.path.isfile(sleap_output_R):
                os.remove(sleap_output_R)
            if os.path.isfile(sleap_output_L):
                os.remove(sleap_output_L)
	
	# prepare sleap
        prepare_sleap(trial_dirs)
        
        # run sleap on right side
        camera_num = "5" 
        run_sleap(camera_num)        
        # rename body parts in sleap_output.h5 to add 'R_'
        for trial_dir in trial_dirs:
            sleap_output = os.path.join(trial_dir, "behData", "images", "sleap_output.h5")
            with h5py.File(sleap_output, 'r+') as file:
                nodes = file['node_names'][:]
                for i, node in enumerate(nodes):
                    node_str = node.decode('utf-8')
                    node_str = 'R_' + node_str
                    nodes[i] = node_str.encode('utf-8')
                file['node_names'][:] = nodes
                    
        # transfer to df file
        for trial_dir in trial_dirs:
            beh_df = os.path.join(trial_dir, load.PROCESSED_FOLDER, global_params.df3d_df_out_dir)
            beh_df = add_sleap_to_beh_df(trial_dir=trial_dir, beh_df=beh_df, out_dir=beh_df)
            
        # rename sleap output file for R side  
        for trial_dir in trial_dirs:  
            sleap_output = os.path.join(trial_dir, "behData", "images", "sleap_output.h5")
            sleap_output_R = os.path.join(trial_dir, "behData", "images", "sleap_output_R.h5")
            os.rename(sleap_output, sleap_output_R)
        
        for trial_dir in trial_dirs:
            # run sleap on left side
            camera_num = "1" 
            run_sleap(camera_num)
            # rename body parts in sleap_output.h5 to add 'L_'
            with h5py.File(sleap_output, 'r+') as file:
                nodes = file['node_names'][:]
                for i, node in enumerate(nodes):
                    node_str = node.decode('utf-8')
                    node_str = 'L_' + node_str
                    nodes[i] = node_str.encode('utf-8')
                file['node_names'][:] = nodes
                
        # transfer to df file
        for trial_dir in trial_dirs:
            beh_df = os.path.join(trial_dir, load.PROCESSED_FOLDER, global_params.df3d_df_out_dir)
            print(beh_df)
            beh_df = add_sleap_to_beh_df(trial_dir=trial_dir, beh_df=beh_df, out_dir=beh_df)
        
        # rename sleap output file for L side   
        for trial_dir in trial_dirs: 
            sleap_output = os.path.join(trial_dir, "behData", "images", "sleap_output.h5")
            sleap_output_L = os.path.join(trial_dir, "behData", "images", "sleap_output_L.h5")
            os.rename(sleap_output, sleap_output_L)
        
        return True
        
class Df3dTask(Task):
    """
    Task to run pose estimation using DeepFly3D and DF3D post processing
    and save results in behaviour dataframe.
    """
    def __init__(self, prio: int=0) -> None:
        super().__init__(prio)
        self.name = "df3d"
        self.previous_tasks = [BehDataTransferTask(), SyncDataTransfer()]

    def test_todo(self, fly_dict: dict) -> bool:
        TODO1 = self._test_todo_trials(fly_dict, file_name=global_params.df3d_df_out_dir)
        trial_dirs_todo = get_selected_trials(fly_dict)
        TODO2 = not all([os.path.isdir(os.path.join(trial_dir, "behData", "images", "df3d"))
                         for trial_dir in trial_dirs_todo])
        if not TODO2:
            found_files = [utils.find_file(os.path.join(trial_dir, "behData", "images", "df3d"),
                                       name="aligned_pose__*.pkl",
                                       raise_error=False) for trial_dir in trial_dirs_todo]
            TODO2 = any([found_file is None for found_file in found_files])
            if not TODO2:
                TODO2 = not all([bool(len(found_file)) is None for found_file in found_files])

        return TODO1 or TODO2

    def run(self, fly_dict: dict, params: PreProcessParams=None) -> bool:
        if not self.wait_for_previous_task(fly_dict):
            return False
        else:
            self.send_status_email(fly_dict)
            print(f"{time.ctime(time.time())}: starting {self.name} task for fly {fly_dict['dir']}")

        self.params = deepcopy(params) if params is not None else deepcopy(global_params)
        self.params.overwrite = fly_dict["overwrite"]

        trial_dirs = get_selected_trials(fly_dict)

        print("STARTING PREPROCESSING OF FLY: \n" + fly_dict["dir"])
        preprocess = PreProcessFly(fly_dir=fly_dict["dir"], params=self.params,
                                   trial_dirs=trial_dirs)
        preprocess._pose_estimate()
        preprocess._post_process_pose()
        return True

class VideoTask(Task):
    """
    Task to make videos for each trial that contain one behavioural camera, DFF,
    and registered raw data.
    Additionaly copy the videos to a central folder
    """
    def __init__(self, prio=0):
        super().__init__(prio)
        self.name = "video"
        self.previous_tasks = [DffTask(), SyncDataTransfer()]

    def test_todo(self, fly_dict):
        trial_dirs = get_selected_trials(fly_dict)
        TODO = False
        for trial_dir in trial_dirs:
            trial_name = "_".join(trial_dir.split(os.sep)[-3:])
            file_name = trial_name+"_"+global_params.dff_beh_video_name+".mp4"
            TODO = TODO or not os.path.isfile(os.path.join(trial_dir, load.PROCESSED_FOLDER,
                                                           file_name))
            TODO = TODO or not os.path.isfile(os.path.join(CURRENT_USER["video_dir"], file_name))
        return TODO

    def run(self, fly_dict, params=None):
        if not self.wait_for_previous_task(fly_dict):
            return False
        else:
            self.send_status_email(fly_dict)
            print(f"{time.ctime(time.time())}: starting {self.name} task for fly {fly_dict['dir']}")

        self.params = deepcopy(params) if params is not None else deepcopy(global_params)

        trial_dirs = get_selected_trials(fly_dict)

        self.params.dff_video_max_length = None # 100
        self.params.dff_video_downsample = 2
        self.params.overwrite = self.params.overwrite = fly_dict["overwrite"]

        preprocess = PreProcessFly(fly_dir=fly_dict["dir"], params=self.params,
                                   trial_dirs=trial_dirs)
        for i, trial_dir in enumerate(trial_dirs):
            preprocess._make_dff_behaviour_video_trial(i_trial=i, mask=None, include_2p=True)
            shutil.copy2(
                os.path.join(trial_dir, load.PROCESSED_FOLDER,
                f"{preprocess.date}_{preprocess.genotype}_Fly{preprocess.fly}_" +\
                f"{preprocess.trial_names[i]}_{preprocess.params.dff_beh_video_name}.mp4"),
                CURRENT_USER["video_dir"])

        return True

class LaserStimProcessTask(Task):
    """
    Task to process the synchronisation signals from laser stimulation
    """
    def __init__(self, prio=0):
        super().__init__(prio)
        self.name = "laser_stim_process"
        self.previous_tasks = ["OR", FictracTask(), WheelTask(), DfTask()]

    def test_todo(self, fly_dict):
        TODO1 = self._test_todo_trials(fly_dict, file_name="stim_paradigm.pkl")
        TODO2 = self._test_todo_trials(fly_dict, file_name=global_params.df3d_df_out_dir)
        return TODO1 or TODO2

    def run(self, fly_dict, params=None):
        if not self.wait_for_previous_task(fly_dict):
            return False
        else:
            self.send_status_email(fly_dict)
            print(f"{time.ctime(time.time())}: starting {self.name} task for fly {fly_dict['dir']}")

        self.params = deepcopy(params) if params is not None else deepcopy(global_params)
        self.params.overwrite = fly_dict["overwrite"]

        trial_dirs = get_selected_trials(fly_dict)

        for trial_dir in trial_dirs:
            beh_df = os.path.join(trial_dir, load.PROCESSED_FOLDER, self.params.df3d_df_out_dir)
            
            _ = get_sync_signals_stimulation(trial_dir,
                                             sync_out_file="stim_sync.pkl",
                                             paradigm_out_file="stim_paradigm.pkl",
                                             overwrite=self.params.overwrite,
                                             index_df=beh_df,
                                             df_out_dir=beh_df)
            
            twop_df = os.path.join(trial_dir, load.PROCESSED_FOLDER, self.params.twop_df_out_dir)
            # change self.previous_task temporarily to only persue behavioural data processing
            # if there actually is behavioural data
            _ = get_beh_info_to_twop_df(beh_df, twop_df, twop_df_out_dir=twop_df)
            previous_tasks = deepcopy(self.previous_tasks)
            self.previous_tasks = ["OR", FictracTask(), WheelTask()]
            if os.path.isfile(twop_df) and self.test_previous_task_ready(fly_dict):
                _ = add_beh_state_to_twop_df(twop_df, twop_df_out_dir=twop_df)
            self.previous_tasks = previous_tasks
        return True

class OlfacStimProcessTask(Task):
    """
    Task to process the synchronisation signals from olfactory stimulation
    """
    def __init__(self, prio=0):
        super().__init__(prio)
        self.name = "olfac_stim_process"
        self.previous_tasks = ["OR", FictracTask(), WheelTask(), DfTask()]

    def test_todo(self, fly_dict):
        TODO1 = self._test_todo_trials(fly_dict, file_name="olfac_paradigm.pkl")
        TODO2 = self._test_todo_trials(fly_dict, file_name=global_params.df3d_df_out_dir)
        return TODO1 or TODO2

    def run(self, fly_dict, params=None):
        if not self.wait_for_previous_task(fly_dict):
            return False
        else:
            self.send_status_email(fly_dict)
            print(f"{time.ctime(time.time())}: starting {self.name} task for fly {fly_dict['dir']}")

        self.params = deepcopy(params) if params is not None else deepcopy(global_params)
        self.params.overwrite = fly_dict["overwrite"]

        trial_dirs = get_selected_trials(fly_dict)

        for trial_dir in trial_dirs:
            beh_df = os.path.join(trial_dir, load.PROCESSED_FOLDER, self.params.df3d_df_out_dir)
            
            _ = get_sync_signals_olfaction(trial_dir,
                                           sync_out_file="olfac_sync.pkl",
                                           paradigm_out_file="olfac_paradigm.pkl",
                                           overwrite=self.params.overwrite,
                                           index_df=beh_df,
                                           df_out_dir=beh_df,
                                           new_olfac=True,
                                           new_olfac_odour="H2O",  # TODO: make more general for multiple odours
                                           stim_time=5)  # TODO: make this more general
            
            twop_df = os.path.join(trial_dir, load.PROCESSED_FOLDER, self.params.twop_df_out_dir)
            if os.path.isfile(twop_df):
                _ = get_beh_info_to_twop_df(beh_df, twop_df, twop_df_out_dir=twop_df)
                _ = add_beh_state_to_twop_df(twop_df, twop_df_out_dir=twop_df)
            
        return True

class VideoPlot3DTask(Task):
    """
    Task to make fly through videos and projection plots for averaged 3d volumes
    """
    def __init__(self, prio=0):
        super().__init__(prio)
        self.name = "video_plot_3d"
        self.previous_tasks = [TifTask()]

    def test_todo(self, fly_dict):
        TODO1 = self._test_todo_trials(fly_dict, file_name="zstack.mp4") or \
            self._test_todo_trials(fly_dict, file_name="ystack.mp4") or \
            self._test_todo_trials(fly_dict, file_name="xstack.mp4")
        TODO2 = self._test_todo_trials(fly_dict, file_name="3dproject.png")
        return TODO1 or TODO2

    def run(self, fly_dict, params=None):
        if not self.wait_for_previous_task(fly_dict):
            return False
        else:
            self.send_status_email(fly_dict)
            print(f"{time.ctime(time.time())}: starting {self.name} task for fly {fly_dict['dir']}")

        self.params = deepcopy(params) if params is not None else deepcopy(global_params)

        trial_dirs = get_selected_trials(fly_dict)

        self.params.overwrite = self.params.overwrite = fly_dict["overwrite"]
        preprocess = PreProcessFly(fly_dir=fly_dict["dir"], params=self.params,
                                   trial_dirs=trial_dirs)

        for i, processed_dir in enumerate(preprocess.trial_processed_dirs):
            green_tif = os.path.join(processed_dir, self.params.green_raw)
            green_avg = os.path.join(processed_dir,"green_avg.tif") if os.path.isfile(green_tif) else None

            red_tif = os.path.join(processed_dir, self.params.red_raw)
            red_avg = os.path.join(processed_dir,"red_avg.tif") if os.path.isfile(red_tif) else None
            show3d.make_avg_videos_3d(
                green=green_tif if os.path.isfile(green_tif) else None,
                red=red_tif if os.path.isfile(red_tif) else None,
                green_avg=green_avg,
                red_avg=red_avg,
                out_dir=processed_dir
            )
            show3d.plot_projections_3d(
                green=green_tif if os.path.isfile(green_tif) else None,
                red=red_tif if os.path.isfile(red_tif) else None,
                green_avg=green_avg,
                red_avg=red_avg,
                out_dir=processed_dir
            )

            shutil.copy2(
                os.path.join(processed_dir, "zstack.mp4"),
                os.path.join(CURRENT_USER["video_dir"],
                             f"{preprocess.date}_{preprocess.genotype}_" +\
                             f"Fly{preprocess.fly}_{preprocess.trial_names[i]}_zstack.mp4")
            )
            shutil.copy2(
                os.path.join(processed_dir, "ystack.mp4"),
                os.path.join(CURRENT_USER["video_dir"],
                             f"{preprocess.date}_{preprocess.genotype}_" +\
                             f"Fly{preprocess.fly}_{preprocess.trial_names[i]}_ystack.mp4")
            )
            shutil.copy2(
                os.path.join(processed_dir, "xstack.mp4"),
                os.path.join(CURRENT_USER["video_dir"],
                             f"{preprocess.date}_{preprocess.genotype}_" +\
                             f"Fly{preprocess.fly}_{preprocess.trial_names[i]}_xstack.mp4")
            )
            shutil.copy2(
                os.path.join(processed_dir, "3dproject.png"),
                os.path.join(CURRENT_USER["video_dir"],
                             f"{preprocess.date}_{preprocess.genotype}_" +\
                             f"Fly{preprocess.fly}_{preprocess.trial_names[i]}_3dproject.png")
            )

        return True

task_collection = {
    "twop_data_transfer": TwopDataTransferTask(prio=-100),
    "beh_data_transfer": BehDataTransferTask(prio=-100),
    "pre_cluster": PreClusterTask(prio=10),
    "pre_cluster_green_only": PreClusterGreenOnlyTask(prio=10),
    "tif": TifTask(prio=9),
    "cluster": ClusterTask(prio=-100),
    "post_cluster": PostClusterTask(prio=-5),
    "denoise": DenoiseTask(prio=-6),
    "dff": DffTask(prio=-10),
    "summary_stats": SummaryStatsTask(prio=-16),
    "prepare_roi_selection": PrepareROISelectionTask(prio=-17),
    "roi_selection": ROISelectionTask(prio=-100),
    "roi_signals": ROISignalsTask(prio=-18),
    "df": DfTask(prio=0),
    "fictrac": FictracTask(prio=0),
    "wheel": WheelTask(prio=0),
    "sleap": SleapTask(prio=-2),
    "sleapRL": SleapRLTask(prio=-2),
    "df3d": Df3dTask(prio=-20),
    "video": VideoTask(prio=-19),
    "laser_stim_process": LaserStimProcessTask(prio=-1),
    "olfac_stim_process": OlfacStimProcessTask(prio=-1),
    "video_plot_3d": VideoPlot3DTask(prio=-15)
}
