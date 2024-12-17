"""
task manager module housing the TaskManager class
TaskManager: class to prioritise and sequentially execute different tasks

contains functions:
    - _create_torun_table: reads the supplied text file and creates the torun_table from the list of trials/tasks to process
        (used during initialization of the TaskManager
    - run: run the tasks manager to sequentially process all toruns from self.torun_dicts
    
also contains additional supporting functions
"""

from copy import deepcopy
from typing import List, Dict
import time
from datetime import datetime
import numpy as np
import pandas as pd
import os
import itertools

from imabeh.run.tasks import Task, task_collection, pipeline_dict

from imabeh.run.userpaths import LOCAL_DIR, user_config # get the current user configuration (paths and settings)

from imabeh.run.logmanager import LogManager
from imabeh.run.flytablemanager import FlyTableManager


class TaskManager():
    """
    class to prioritise and sequentially execute different tasks
    """
    def __init__(self,
        log: LogManager,
        user_config: dict = user_config,
        task_collection: Dict[str, Task] = task_collection
    ) -> None:
        """
        Class to prioritise and sequentially execute different tasks.
        Most important component is a table of tasks torun that is managed: self.torun_table
        A "torun" ITEM IS DEFINED AS A SINGLE TASK TO BE RUN ON A SINGLE TRIAL
        Each torun item (row in the table) will have the following fields:
            - index: the index of the torun in the order of tasks to run 
            - dir: the base directory of the fly, where the data is stored
            - trial: the fly trial to analyze
            - full_path: the full path to the trial directory (dir/trial)
            - task: the name of the task to be done. Later used to index in self.task_collection
            - overwrite: whether or not to force an overwrite of the previous results
            - status: whether the to_run is "ready", "running", or "waiting"

        Parameters
        ----------
        task_collection : Dict[str, Task]
            a dictionary indexed by strings containing instances of the class Task.
            This dictionary should contain all possible tasks.
            Will be used to check that all tasks are valid.
        user_config : dict
            dictionary with user specific parameters, such as file directories, etc.
        log : LogManager
        """

        # set base properties
        self.user_config = user_config
        self.task_collection = task_collection
        self.txt_file_to_process = self.user_config["txt_user_and_dirs_to_process"]
        self.txt_file_running = self.user_config["txt_file_running"]
        # set time to wait between recurrently checking for non-python/shell tasks to run
        self.t_wait_s = 60

        # get the fly_table (to check tasks that have previously been run)
        self.fly_table = FlyTableManager()

        # torun_table: table of tasks to run on trials (toruns)
        # each row is a torun, with columns: fly_dir, trial, task, overwrite, status
        self._create_torun_table(log)
   
    @property
    def n_toruns(self) -> int:
        """
        get the number of torun items as rows in self.torun_table

        Returns
        -------
        int
        """
        return len(self.torun_table)


    ## MAIN FUNCTIONS

    def _create_torun_table(self, log: LogManager):
        """
        reads the supplied text file and 
        creates the torun_table from the list of trials/tasks to process
        USED ONLY ONCE AT START (.__init__)

        For each trial in the trials_to_process list:

        checks that the trial dir exists
        checks if tasks exist in task_collection
        checks if tasks have been completed previously. If so, checks if they need to be overwritten.
            if already completed and overwrite = False, log and remove from list of to_run
            if already completed and overwrite = True, change status in fly_table to 2 
                to allow re-running and for tasks to depende on it to wait until it is re-run
        finally checks the pre-requisites for each task and set statuses acordingly

        Parameters
        ----------
        log: LogManager to log the creation of the torun_table

        Generates
        -------
        self.torun_table with the following columns, containing the toruns from self.trials_to_process:
            - "fly_dir": the base directory of the fly
            - "trial": which trial to run on
            - "full_path": the full path to the trial directory (dir/trial)
            - "task": the name of the task to run
            - "overwrite": whether or not to force an overwrite of the previous results
            - "status": whether the to_run is "ready", "running", or "waiting"
            - "taskstatus_log": the path to the taskstatus log file
        """
        # log the start of the creation of the torun_table
        log.add_line_to_log("-------CREATING TORUN TABLE-------\n")

        # create empty to_run table
        header = ["fly_dir", "trial", "full_path", "task", "overwrite", "status", "taskstatus_log"]
        self.torun_table = pd.DataFrame(columns=header)

        # read the text file and get the list of flies/trials/tasks to process
        # all the modifications (pipelines, keywords etc. are processed here)
        fly_dicts = self._read_fly_dirs(self.txt_file_to_process, log) 

        # fill with new toruns from the fly_dicts
        # checks that fly dirs exist and that tasks are valid (in task_collection) - log if not
        for trial_dict in fly_dicts:
            self._add_toruns_from_trial(trial_dict, log)

        # check if there are duplicate toruns
        # if so, remove duplicates - prioritise the first one in list
        self._check_duplicates(log)

        # iterate over toruns to check if tasks have already been completed/need to be overwritten
        for torun_index, torun in self.torun_table.iterrows():

            # convert torun row to dict
            torun_dict = torun.to_dict()

            # check if any tasks have been completed previously (without errors) using the fly table
            table_status = self.fly_table.check_trial_task_status(torun_dict)

            # if already completed and overwrite = False, log and remove from list of to_run
            if table_status == 1 and torun.overwrite == False: # 1 = done
                log.add_line_to_log(f"ALREADY DONE - task '{torun.task}' for fly '{torun.fly_dir}' trial '{torun.trial}' already completed \n")
                self._remove_torun(torun_dict)

            # if already completed and overwrite = True, change status in fly_table to 2 (to allow it to be re-run)
            elif table_status == 1 and torun.overwrite == True: 
                log.add_line_to_log(f"OVERWRITE - task '{torun.task}' for fly '{torun.fly_dir}' trial '{torun.trial}' already completed but will be overwritten /n")
                self.fly_table.update_trial_task_status(torun_dict, status = 2)
        
        # iterate over toruns to check pre-requisites for each task and set statuses acordingly
        # "ready" if all pre-requisites are met, "waiting" otherwise
        # also remove if prerequisites are missing
        for torun_index, torun in self.torun_table.iterrows():
            self._check_prerequisites(torun, torun_index, log)

        # log the creation of the torun_table
        log.add_line_to_log("\n-------TORUN TABLE CREATED-------\n")

    def run(self, log) -> None:
        """
        run the tasks manager to sequentially process all toruns from self.torun_dicts.

        The main loop will:
        - check if there are tasks left to run - if not, finish
        - check if there are any running tasks (non-python/bash tasks), and if so whether they have finished
        - check if any tasks are ready to run (all prerequisites met)
            - if none are ready, wait and check again (again, for non-python/bash tasks)
            - if any are ready, run the next ready task
        """
        # log the start of the task manager
        log.add_line_to_log("\n-------TASK MANAGER STARTED-------\n")

        # check if there are tasks left to run
        while self.n_toruns:

            # check all prereqs and update status/remove todos
            for torun_index, torun in self.torun_table.iterrows():
                self._check_prerequisites(torun, torun_index, log)

            # checks all "running" tasks to see if any have finished - correctly (1) or with errors (2)
            # this will only be necessary for tasks that are run outside of python/bash (ex. on the cluster)
            running_tasks = self.torun_table[self.torun_table['status'] == "running"]
            if not running_tasks.empty:
                self._check_running_tasks(running_tasks, log)

            # check if any tasks are ready to run
            ready_tasks = self.torun_table[self.torun_table['status'] == "ready"]

            # if none are ready, wait and check again
            if ready_tasks.empty:
                log.add_line_to_log('Waiting... no tasks ready to run \n')
                time.sleep(self.t_wait_s)
                print(self.torun_table)

            # otherwise, run the next ready task
            else:
                self._execute_next_task(log)

        # if no tasks are left, log and finish
        log.add_line_to_log("\n-------TASK MANAGER FINISHED-------")

        
    ## SUPPORTING FUNCTIONS - for create_torun_table > _read_fly_dirs

    def _read_fly_dirs(self, txt_file: str, log) -> List[dict]:
        """
        reads the supplied text file and returns a list of dictionaries with information for each fly trial to process.
        USED ONLY ONCE AT START (.__init__)

        General requested format of a line in the txt file: fly_dir||trial1,trial2||task1,task2,!task3
        example (see _user_and_fly_dirs_to_process_example.txt for more):
        date_genotype/Fly1||001_beh,002_beh||fictrac,df3d

        Modifications:
            instead of the full trial name, the begginning of the trial name is enough (usually the trial number)
            'all' instead of the trial name will fetch all trials in the fly dir (unless excluded using e-)
            k- before a trial indicates trial keyword (all trials with that keyword will be included, unless excluded using e-)
            e- before a trial name (or start of name) indicates this trial will be excluded

            ! before a task forces an overwrite.
            p- before a task indicates pipeline (a preset set of tasks)

        Parameters
        ----------
        txt_file : str, optional
            location of the text file
        log: logManager
            to log any issues with the trials/pipelines, exclusions...

        Returns
        -------
        fly_dicts: List[dict]
            list of trial dict with the following fields for each fly trial:
            - "fly_dir": str - the base directory of the fly
            - "trials": List[str] - which trials to run on
            - "tasks": List[str] - names of the tasks to run
        """

        # read file
        with open(txt_file) as file:
            lines = file.readlines()
            lines = [line.rstrip() for line in lines]

        # get list of fly trials/tasks, splitting tasks and trials strings
        fly_dicts = []
        for line in lines:
            # ignore commented lines
            if line.startswith("#") or line == "" or line.startswith("CURRENT_USER"):
                continue
            # split into sections (fly_dir,trials,tasks)
            strings = line.split("||")
            # split trials and tasks to make a list of each
            trials = strings[1].split(',')
            tasks = strings[2].split(',')
            # make fly_dict
            fly_dict = {
                "fly_dir": strings[0],
                "trials": trials,
                "tasks": tasks,
                }
            fly_dicts.append(fly_dict)

        # iterate across fly_dicts to check for modifications
        # (in reverse order so elements can be removed without affecting the loop)
        for f, fly_dict in reversed(list(enumerate(fly_dicts))): 

            # check that all fly_dirs are real folders - if not, log and remove
            fly_dir = os.path.join(user_config["labserver_data"],fly_dict['fly_dir'])
            if not os.path.isdir(fly_dir):
                log.add_line_to_log(f"Fly dir '{fly_dir}' does not exist\n")
                fly_dicts.pop(f)
                continue

            # replace pipelines with the tasks they contain
            # if p-!, add ! in front of all tasks (each will be overwritten)
            fly_dict = self._read_pipelines(fly_dict, log)

            # get all trials if 'all' is present
            fly_dict = self._get_all_trials(fly_dict)

            # get trials based on keyworkds (k-)
            fly_dict = self._read_keywords(fly_dict, log)

            # get full trial names if needed
            fly_dict = self._get_full_trial(fly_dict, log)

            # exclude any specified trials (or starts of trials)
            fly_dict = self._exclude_trials(fly_dict, log)

            # replace in fly_dicts
            fly_dicts[f] = fly_dict

        return fly_dicts

    def _read_pipelines(self, fly_dict, log):
        """ If any task starts with 'p-', replace it with the tasks in the pipeline.
        The list of preset pipelines is imported from imabeh/run/tasks.py

        If the pipeline is set to be overwritten (p-!), add ! in front of all tasks in the pipeline

        Parameter and Returns
        ---------------------
        fly_dict: dict 
            dict containing fly_dir, trials, tasks
        log: LogManager
            to log any wrong pipelines entered
        """
        tasks = fly_dict['tasks']
        for t, task in reversed(list(enumerate(tasks))): # reverse order so elements can be removed without affecting the loop
            if task.startswith("p-"):
                # get pipeline name and remove ! if present
                pipeline = task[2:].replace("!","")

                # get tasks in pipeline
                try:
                    new_tasks = pipeline_dict[pipeline]
                # if pipeline not found in pipeline_dict, log and remove
                except KeyError:
                    log.add_line_to_log(f"Pipeline '{pipeline}' is not defined in pipeline_dict\n")
                    tasks.pop(t)
                    continue

                # if overwrite (p-!), add ! in front of all tasks
                if task.startswith("p-!"):
                    new_tasks = ["!" + t for t in new_tasks]
                
                # replace pipeline with new tasks
                tasks[t:t+1] = new_tasks  

        # add to fly_dict and return
        fly_dict['tasks'] = tasks
        return fly_dict

    def _get_all_trials(self, fly_dict):
        """ If any trial is 'all', search the fly_dir and add all folders found as trials 

        Parameter and Returns
        ---------------------
        fly_dict: dict 
            dict containing fly_dir, trials, tasks
        """
        trials = fly_dict['trials']
        for t, trial in enumerate(trials):
            if trial == 'all':
                # get list of folders (trials) within fly_dir and replace 'all' with trials
                fly_dir = os.path.join(user_config["labserver_data"],fly_dict['fly_dir'])
                new_trials = [f for f in os.listdir(fly_dir) if os.path.isdir(os.path.join(fly_dir, f))]
                trials[t:t+1] = new_trials
        
        fly_dict['trials'] = trials
        
        return fly_dict
    
    def _read_keywords(self, fly_dict, log):
        """ If any trial starts with k-, search the fly_dir and add all folders found as trails 
        if they contain the keywork that follows.

        Parameter and Returns
        ---------------------
        fly_dict: dict 
            dict containing fly_dir, trials, tasks
        log: LogManager
            to log any wrong keywords that don't match any trials
        """
        trials = fly_dict['trials']
        for t, trial in reversed(list(enumerate(trials))): # reverse order so elements can be removed without affecting the loop
            if trial.startswith('k-'):
                keyword = trial[2:]
                # get list of folders (trials) within fly_dir - but only if they contain the keyword
                fly_dir = os.path.join(user_config["labserver_data"],fly_dict['fly_dir'])
                new_trials = [f for f in os.listdir(fly_dir) 
                            if os.path.isdir(os.path.join(fly_dir, f)) 
                            and keyword in f
                            ]
                # if not matching trials were found, remove and log
                if len(new_trials) == 0:
                    log.add_line_to_log(f"No trials match keyword '{keyword}' in fly_dir {fly_dir}\n")
                    trials.pop(t)
                else:
                    trials[t:t+1] = new_trials

        # add back to fly_dict and return
        fly_dict['trials'] = trials
        return fly_dict

    def _get_full_trial(self, fly_dict, log):
        """ Instaed of full trial names, the start of the name is sufficient.
        Check whether each trial is a real dir within fly_dir, and if not,
        search for any folder that starts with the trial given.
        If no folder matches or more than one is found, log and remove.

        Parameter and Returns
        ---------------------
        fly_dict: dict 
            dict containing fly_dir, trials, tasks
        log: LogManager
            to log any mistakes in trials
        """
        trials = fly_dict['trials']

        # get list of folders (real trials) within fly_dir
        fly_dir = os.path.join(user_config["labserver_data"],fly_dict['fly_dir'])
        real_trials = [f for f in os.listdir(fly_dir) if os.path.isdir(os.path.join(fly_dir, f))]

        for t, trial in reversed(list(enumerate(trials))): # reverse order so elements can be removed without affecting the loop
            # check if trial is contained within real_trials (as is not to be excluded)
            if trial not in real_trials and not trial.startswith('e-'):
                # check if any real_trials start with trial
                trial_match = [t for t in real_trials if t.startswith(trial)]
                # if only one match, replace
                if len(trial_match) == 1:
                    trials[t] = trial_match[0]
                # if no matches or more than one, log and remove
                else:
                    log.add_line_to_log(f"Trial {trial} has {len(trial_match)} matches in {fly_dir}\n")
                    trials.pop(t)

        # add to fly_dict and return
        fly_dict['trials'] = trials
        return fly_dict
        
    def _exclude_trials(self, fly_dict, log):
        """ If any trial starts with e-, find match in trial list and remove.
        If no match (or too many) is found, log and ignore.

        Parameter and Returns
        ---------------------
        fly_dict: dict 
            dict containing fly_dir, trials, tasks
        log: LogManager
            to log any trial that could not be excluded (not found in fly_dict)
        """
        trials = fly_dict['trials']
        for t, trial in enumerate(trials):
            if trial.startswith('e-'):
                exclude = trial[2:]

                # remove e- trial itself
                trials.pop(t)

                # find any other matching trials to exclude
                match = [t for t in trials if t.startswith(exclude)]
                # if no matches are found, ignore
                if len(match) == 0:
                    log.add_line_to_log(f"Excluded trial {exclude} has NO matches in fly_dir {fly_dict['fly_dir']}. IGNORED \n")
                # otherwise, exclude all matches
                else:
                    trials = [t for t in trials if not t.startswith(match[0])]
                    # warn in log if more than one match
                    if len(match) > 1:
                        log.add_line_to_log(f"Excluded trial {exclude} has {len(match)} matches in fly_dir {fly_dict['fly_dir']}. ALL WILL BE EXCLUDED \n")

        # add to fly_dict and return
        fly_dict['trials'] = trials
        return fly_dict


    ## SUPPORTING FUNCTIONS - for create_torun_table > _add_toruns_from_trial

    def _add_toruns_from_trial(self, trial_dict : dict, log):
        """
        function to append new toruns as rows to self.torun_table from a trial_dict
        USED ONLY ONCE AT START (.__init__)

        Checks that fly dirs exist and that tasks are valid (in task_collection).
        Will also check if task needs to be overwritten (! at start of task name).
        Adds an index to indicate the order in which the tasks must be run - 
        default order is as provided in the text file (could add a function to reorder, if needed at some point)

        Parameters
        ----------
        trial_dict: dict
            trial dict with the following fields for each fly trial:
            - "fly_dir": the base directory of the fly
            - "trials": List[str] - which trials to run on
            - "tasks": List[str] - names of the tasks to run
        log: LogManager to log any errors

        Generates
        -------
            new rows in self.torun_table, one for each task in the trial_dict
        """
        # get last torun order in table
        order = self.torun_table.index.max()
        if np.isnan(order): #(no toruns yet)
            order = -1
            
        # for each trial and task, make a new torun dict (will become a row)
        for trial_name, task_name in itertools.product(trial_dict["trials"], trial_dict["tasks"]):
            # create new torun dict
            new_torun = deepcopy(trial_dict)
            new_torun["trial"] = trial_name
            new_torun["task"] = task_name
            main_path = user_config["labserver_data"]
            new_torun["full_path"] = os.path.join(main_path,new_torun['fly_dir'],trial_name)

            # check that fly dir exists - if not, move on to next torun
            if not os.path.exists(new_torun["full_path"]):
                log.add_line_to_log(f"Fly directory {new_torun['full_path']} does not exist \n")
                break

            # check if tasks must be overwritten
            if task_name.startswith("!"):
                task_name = task_name[1:]
                new_torun["task"] = task_name
                new_torun["overwrite"] = True
            else:
                new_torun["task"] = task_name
                new_torun["overwrite"] = False

            # check that task exists in task_collection - if not, move on to next task
            if task_name not in self.task_collection.keys():
                log.add_line_to_log(f"Task '{task_name}' is not defined in task_collection \n")
                continue

            # set status to ready (default) and taskstatus_log to None
            new_torun["status"] = "ready"
            new_torun["taskstatus_log"] = None  
           
            # add torun to table as new row
            order += 1
            self.torun_table.loc[order] = (new_torun)
        
        # sort indexes (order)
        self.torun_table = self.torun_table.sort_index()

    def _find_torun_in_table(self, torun_dict : dict) -> int:
        """
        find the index of a torun in the self.torun_table
        USED IN MANY OTHER FUNCTIONS

        Parameters
        ----------
        torun_dict: dict
            torun dict with the following fields:
            - "fly_dir": the base directory of the fly
            - "trial": which trial to run on
            - "task": the name of the task to run

        Returns
        -------
        row.index : int
            index of the torun in the torun_table
        """

        # find row that matches torun_dict
        row = self.torun_table.loc[
            (self.torun_table["fly_dir"] == torun_dict["fly_dir"]) &  
            (self.torun_table["trial"] == torun_dict["trial"]) & 
            (self.torun_table["task"] == torun_dict["task"])
        ]
        # if no match, return None
        if row.empty:
            return None
        else:
            # return index
            return row.index
    
    def _remove_torun(self, torun_dict : dict):
        """
        function to remove a torun row from the self.torun_table
        USED AT START IF TASK HAS BEEN COMPLETED PREVIOUSLY AND OVERWRITE = FALSE
        ALSO USED AFTER A TASK HAS BEEN COMPLETED

        Parameters
        ----------
        torun_dict: dict
            torun dict with the following fields:
            - "fly_dir": the base directory of the fly
            - "trial": which trial to run on
            - "task": the name of the task to run
        Generates
        -------
            removes the row from self.torun_table
        """

        # find row index that matches torun_dict
        torun_index = self._find_torun_in_table(torun_dict)
    
        # remove row
        self.torun_table = self.torun_table.drop(torun_index)

        
    def _check_duplicates(self, log: LogManager):
        """ 
        function to check for duplicates in the torun_table and remove them
        priority is given to the first torun in the list (in case overwritting is different)
        USED AT START TO CLEAN UP THE torun_table (.__init__)

        Parameters
        ----------
        log: LogManager to log the removal of duplicates

        Generates
        -------
            removes duplicates from the torun_table
        """
        # set columns to consider when checking for duplicates
        columns_to_check = ["fly_dir", "trial", "task"]
        # find duplicates
        duplicates_exist = self.torun_table.duplicated(subset=columns_to_check).any()

        # if duplicates exist, log and remove
        if duplicates_exist:
            log.add_line_to_log("WARNING: Duplicate toruns found in torun_table. Removing duplicates \n")
            self.torun_table = self.torun_table.drop_duplicates(subset=columns_to_check)

    def _check_prerequisites(self, torun, torun_index, log : LogManager) -> bool:
        """
        function to check if all the prerequisites for a task are met and set statuses/remove toruns acordingly
        "ready" if all pre-requisites are met, "waiting" otherwise.
        Checks that all incomplete prerequisites for each task are in the torun_table,
        and that they are earlier in the order than the task in question.
            if the prerequisites are missing, log and remove
            if they are present but below the task, log and remove
        WILL BE USED AT START AND BEFORE THE NEXT TASK IS CHOSEN TO RUN

        Parameters
        ----------
        torun : pandas.Series row
            torun table row with the following columns:
            - "fly_dir": the base directory of the fly
            - "trial": which trial to run on
            - "task": the name of the task to run
        torun_index : index of the torun row in the torun_table
        log: LogManager to log any reordering/removal

        Generates
        -------
            changes the status of/removes the torun in the torun_table based on prerequisite status
        """

        # convert torun to dict
        torun_dict = torun.to_dict()
        # get the task
        task = self.task_collection[torun_dict["task"]]()
        # get the prerequisites for the task
        prerequisites = task.prerequisites

        # check if all prerequisite tasks have been completed previously
        prereqs_missing = []
        for prereq in prerequisites:
            prereq_dict = deepcopy(torun_dict)
            prereq_dict["task"] = prereq
            prereq_status = self.fly_table.check_trial_task_status(prereq_dict)

            if prereq_status != 1:
                prereqs_missing.append(prereq)

        # if some prerequisites have not yet been completed, check if they are in the torun_table
        # and if they are before the task in question
        # if not, remove the task and log
        if prereqs_missing:
            all_prereqs_present = True

            for prereq in prereqs_missing:
                # find prereq in torun_table
                prereq_dict = deepcopy(torun_dict)
                prereq_dict["task"] = prereq
                prereq_index = self._find_torun_in_table(prereq_dict)

                # if missing, remove the task and log
                if prereq_index is None:
                    log.add_line_to_log(f"Prerequisite task '{prereq}' for task '{torun_dict['task']}' for fly '{torun_dict['fly_dir']}' trial '{torun_dict['trial']}' is missing. Removing task \n")
                    self._remove_torun(torun_dict)
                    all_prereqs_present = False
                    break

                # if present, check if it is before the task. If not, remove
                elif prereq_index > torun_index:
                    log.add_line_to_log(f"Prerequisite task '{prereq}' for task '{torun_dict['task']}' for fly '{torun_dict['fly_dir']}' trial '{torun_dict['trial']}' is after the task. Removing task\n")
                    self._remove_torun(torun_dict)
                    all_prereqs_present = False 
                    break

            # if all prereqs are present and before the task, set status to "waiting" 
            # (vs default 'ready' for tasks with no incomplete prerequisites)
            if all_prereqs_present:
                self.torun_table.loc[torun_index, "status"] = "waiting"
        
        else:
            # if all prerequisites are met, set status to "ready"
            self.torun_table.loc[torun_index, "status"] = "ready"

    # SUPPORTING FUNCTIONS - for run

    def _check_running_tasks(self, running_tasks, log):
        """
        check if any currently "running" tasks have finished running
        only necessary for tasks that are run outside of python/bash (ex. on the cluster)
        might have to improve whenever a task needs this...

        Parameters
        ----------
        running_tasks: pd.DataFrame
            dataframe with the currently running tasks
        log: LogManager to log the results
        """
        for _, torun in running_tasks.iterrows():
            finished = self._check_task_finished(torun.to_dict())

            # if any previously running tasks finished correctly (1), remove from table and update flyTable
            if finished == 1:
                log.add_line_to_log(f"   Finished '{torun['task']}' task for trial '{torun['fly_dir']}/{torun['trial']}' @ {datetime.now().isoformat(sep=' ')} \n")
                self._remove_torun(torun)
                self.fly_table.update_trial_task_status(torun, status = 1)

            # if any failed (2), remove, update flyTable, and log
            elif finished == 2:
                log.add_line_to_log(f"TASK '{torun['task']}' FOR FLY '{torun['fly_dir']}' TRIAL '{torun['trial']}' FAILED. Removing task \n \n")
                self._remove_torun(torun)
                self.fly_table.update_trial_task_status(torun, status = 2)

    def _check_task_finished(self, torun_dict: dict) -> int:
        """
        check if a currently "running" task has finished running
        This is only necessary for tasks that are run outside of python/bash (ex. on the cluster)
        use each task's own test_finished method to determine if the task has finished
        """

        # get task name
        task_name = torun_dict["task"]
        # get task
        task = self.task_collection[task_name]()
        # get task status
        status = task.test_finished(torun_dict)

        return status

    def _execute_next_task(self, log : LogManager):
        """
        execute the next torun from self.torun_table that is not waiting/running.
        Also update the status of the torun in the torun_table to "running"

        Parameters
        ----------
        log: LogManager to log the execution of the task
        """
        # get first ready task
        next_torun = self.torun_table[self.torun_table['status'] == "ready"].iloc[0]
        torun_index = next_torun.name

        # get task name
        task_name = next_torun["task"]

        # set status to running
        self.torun_table.loc[torun_index, "status"] = "running"

        # initialize and run the task
        # for python tasks, finished = True when successfully finished
        # for others (like cluster tasks), finished = False and you will need to check later using the _check_task_finished method
        task = self.task_collection[task_name]()
        try:
            finished = task.start_run(next_torun.to_dict(), log)

            # if task finished correctly, remove from table and update flyTable
            if finished:
                self._remove_torun(next_torun)
                self.fly_table.update_trial_task_status(next_torun, status = 1)
            # if the task is still running, do nothing here               
        
        #if task failed, remove from table and update flyTable
        except Exception as e:
            self._remove_torun(next_torun)
            self.fly_table.update_trial_task_status(next_torun, status = 2)

