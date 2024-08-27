"""
task manager module housing the TaskManager class
"""
from copy import deepcopy
from typing import List, Dict
import time
import numpy as np
import datetime
import pandas as pd


#from twoppp.run.runutils import read_fly_dirs, read_running_tasks, check_task_running
#from twoppp.run.runutils import send_email, write_running_tasks
from imabeh.run.tasks import Task, task_collection

from imabeh.run.logmanager import LogManager
from imabeh.run.userpaths import LOCAL_DIR, get_current_user_config
from imabeh.run.flytablemanager import FlyTableManager

# get the current user configuration (paths and settings)
user_config = get_current_user_config()


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
        Most important component is a list of tasks todo that is managed: self.todo_dicts
        A "todo" ITEM IS DEFINED AS A SINGLE TASK TO BE RUN ON A SINGLE TRIAL
        Each todo item will have the following fields:
            - index: the index of the todo in the order of tasks to run 
            - dir: the base directory of the fly, where the data is stored
            - trial: the fly trial to analyze
            - task: the name of the task to be done. Later used to index in self.task_collection
            - overwrite: whether or not to force an overwrite of the previous results
            - status: whether the to_run is "ready", "running", "done", or "waiting"

        Parameters
        ----------
        task_collection : Dict[str, Task]
            a dictionary indexed by strings containing instances of the class Task.
            This dictionary should contain all possible tasks
        user_config : dict
            dictionary with user specific parameters, such as file directories, etc.
        log : LogManager
        """

        # set base properties
        self.user_config = user_config
        self.task_collection = task_collection
        self.txt_file_to_process = self.user_config["txt_file_to_process"]
        self.txt_file_running = self.user_config["txt_file_running"]
        # set time to wait between recurrently checking for non-python tasks to run
        self.t_wait_s = 60

        ###########################################################
        self.clean_exit = False

        # get the fly_table
        self.fly_table = FlyTableManager()

        # trials_to_process: list of dicts with info for each fly trial from the txt file
        # one dict per trial, with many tasks possible in each
        self.trials_to_process = []
        self._read_fly_dirs(self.txt_file_to_process)

        # todo_table: table of tasks to run on trials (todos)
        # each row is a todo, with columns: fly_dir, trial, task, overwrite, status
        self._create_todo_table(log)
   
    @property
    def n_todos(self) -> int:
        """
        get the number of todo items as rows in self.todo_table

        Returns
        -------
        int
        """
        return len(self.todo_table)



    def _read_fly_dirs(self, txt_file: str) -> List[dict]:
        """
        reads the supplied text file and returns a list of dictionaries with information for each fly trial to process.
        USED ONLY ONCE AT START
        General requested format of a line in the txt file: fly_dir||trial1,trial2||task1,task2,!task3
        ! before a task forces an overwrite.
        example (see _fly_dirs_to_process_example.txt for more):
        /mnt/nas2/JB/date_genotype/Fly1||001_beh,002_beh||fictrac,!df3d

        Parameters
        ----------
        txt_file : str, optional
            location of the text file

        Returns (saved as self.trials_to_process)
        -------
        trial_dicts: List[dict]
            list of trial dict with the following fields for each fly trial:
            - "fly_dir": the base directory of the fly
            - "trials": which trial to run on
            - "tasks": a comma separated string containing the names of the tasks to run
        """

        # read file
        with open(txt_file) as file:
            lines = file.readlines()
            lines = [line.rstrip() for line in lines]

        # get list of fly trials
        trial_dicts = []
        for line in lines:
            # ignore commented lines
            if line.startswith("#") or line == "":
                continue
            # split into sections (fly_dir,trials,tasks)
            strings = line.split("||")
            # split trials and make a dict for each
            trials = strings[1].split(',')
            for trial in trials:
                trial_dict = {
                    "fly_dir": strings[0],
                    "trial": trial,
                    "tasks": strings[2],
                }
                trial_dicts.append(trial_dict)

        self.trials_to_process = trial_dicts


    def _add_todos_from_trial(self, trial_dict : dict):
        """
        function to append new todos as rows to self.todo_table from a trial_dict
        will check if task needs to be overwritten (! at start)
        add an index to indicate the order in which the tasks must be run - 
        default order is as provided in the text file

        Parameters
        ----------
        trial_dict: dict
            trial dict with the following fields for each fly trial:
            - "fly_dir": the base directory of the fly
            - "trials": which trials to run on
            - "tasks": a comma separated string containing the names of the tasks to run
        """

        # split tasks
        task_list = trial_dict["tasks"].split(',')

        # get last todo order in table
        order = self.todo_table.index.max()
        # check if nan (no todos yet)
        if np.isnan(order):
            order = -1

        # for each task, make a new todo dict (will become a row)
        for task_name in task_list:
            new_todo = deepcopy(trial_dict)

            # check if tasks must be overwritten
            if task_name.startswith("!"):
                new_todo["task"] = task_name[1:]
                new_todo["overwrite"] = True
            else:
                new_todo["task"] = task_name
                new_todo["overwrite"] = False

            # set status to ready (default)
            new_todo["status"] = "ready"
           
            # add todo to table as new row
            order += 1
            self.todo_table.loc[order] = (new_todo)
        
        # sort indexes (order)
        self.todo_table = self.todo_table.sort_index()


    def _remove_todo(self, todo_dict : dict, log: LogManager):
        """
        function to remove a todo row from the self.todo_table

        Parameters
        ----------
        todo_dict: dict
            todo dict with the following fields:
            - "fly_dir": the base directory of the fly
            - "trial": which trial to run on
            - "task": the name of the task to run
        """

       # find row that matches todo_dict
        row = self.todo_table.loc[
            (self.todo_table["fly_dir"] == todo_dict["fly_dir"]) &  
            (self.todo_table["trial"] == todo_dict["trial"]) & 
            (self.todo_table["tasks"] == todo_dict["tasks"])
        ]

        # remove row
        self.todo_table = self.todo_table.drop(row.index)

        # log removal
        LogManager.add_line_to_log(log, f"Removed task {todo_dict['task']} for fly {todo_dict['fly_dir']} trial {todo_dict['trial']} from todo_table.")


    def _create_todo_table(self, log: LogManager):
        """
        create the todo_table from the list of self.trials_to_process
        checks if tasks have been completed previously. If so, checks if they need to be overwritten.
        if already completed and overwrite = False, log and remove from list of to_run
        if already completed and overwrite = True, change status in fly_table to 2 to allow re-running
        but also for tasks to depende on it to wait until it is re-run
        finally checks the pre-requisites for each task and set statuses acordingly
        """
        # create empty table
        header = ["fly_dir", "trial", "task", "overwrite", "status"]
        self.todo_table = pd.DataFrame(columns=header)
        # fill with new todos
        for trial_dict in self.trials_to_process:
            self._add_todos_from_trial(trial_dict)

        # iterate over todos to check task status, overwriting and prerequisites
        for todo_index, todo in self.todo_table.iterrows():
            # convert todo to dict
            todo_dict = todo.to_dict()

            # check if any tasks have been completed previously (without errors) using the fly table
            table_status = self.fly_table.check_trial_task_status(todo_dict)
            # if already completed and overwrite = False, log and remove from list of to_run
            if table_status == 1 and todo.overwrite == False: # 1 = done
                self._remove_todo(todo_dict, log)
            # if already completed and overwrite = True, change status in fly_table to 2
            elif table_status == 1 and todo.overwrite == True: 
                self.fly_table.update_trial_task_status(todo_dict, status = 2)
        
            # check the pre-requisites for each task and set statuses acordingly
            # "ready" if all pre-requisites are met, "waiting" otherwise
            self._check_prerequisites(todo, todo_index)


    def _check_prerequisites(self, todo, todo_index) -> bool:
        """
        check if all the prerequisites for a task are met and set statuses acordingly
        "ready" if all pre-requisites are met, "waiting" otherwise

        Parameters
        ----------
        todo : pandas.Series row
            todo table row with the following columns:
            - "fly_dir": the base directory of the fly
            - "trial": which trial to run on
            - "task": the name of the task to run
        todo_index : index of the todo row in the todo_table
        """

        # convert todo to dict
        todo_dict = todo.to_dict()
        # get the task
        task = self.task_collection[todo_dict["task"]]()
        # get the prerequisites
        prerequisites = task.prerequisites
        # check if all prerequisite tasks have been completed
        prereq_done = True
        for prereq in prerequisites:
            prereq_dict = deepcopy(todo_dict)
            prereq_dict["task"] = prereq
            prereq_status = self.fly_table.check_trial_task_status(todo_dict)
            if prereq_status != 1:
                prereq_done = False
                break
        # set status to "ready" if all prerequisites are met, "waiting" otherwise
        if not prereq_done:
            self.todo_table.loc[todo_index, "status"] = "waiting"
        



    # def execute_next_task(self) -> bool:
    #     """
    #     Execute the next todo from self.todo_dicts that is not waiting.

    #     Returns
    #     -------
    #     bool
    #         True if a task was run to completion, False if all tasks are "waiting.
    #     """

    #     result = False

    #     for i_todo, next_todo in enumerate(self.todo_dicts):
    #         task_name = next_todo["tasks"]
    #         write_running_tasks(next_todo, add=True)
    #         self.todo_dicts[i_todo]["status"] = "running"
    #         try:
    #             result = self.task_collection[task_name].run(fly_dict=next_todo, params=self.params)
    #             if result:
    #                 self.todo_dicts[i_todo]["status"] = "done"
    #                 self.remove_todo(i_todo)
    #                 return True
    #             else:
    #                 self.todo_dicts[i_todo]["status"] = "waiting"
    #                 write_running_tasks(next_todo, add=False)
    #         except Exception as e:
    #             # Log the exception and continue with the next task
    #             print(f"Task {task_name} failed with exception: {e}")
    #             self.todo_dicts[i_todo]["status"] = "failed"
    #             write_running_tasks(next_todo, add=False)
    #             self.log_failed_task(task_name, e)
    #     return False







    # def run(self) -> None:
    #     """
    #     run the tasks manager to sequentially process all todos from self.todo_dicts.
    #     """
    #     try:
    #         # check if there are tasks left to run
    #         while self.n_todos:
    #             # run the next task and check if it works
    #             success = self.execute_next_task()

    #             if not success:
    #                 print('Waiting for pending tasks...')
    #                 # if all tasks are pending wait before checking from the start again
    #                 time.sleep(self.t_wait_s)

    #         self.clean_exit = True

    #     finally:
    #         print("TASK MANAGER clean up: removing tasks from _tasks_running.txt")
    #         for todo in self.todo_dicts:
    #             if todo["status"] == "running":
    #                 write_running_tasks(todo, add=False)
    #         subject = "TASK MANAGER clean exit" if self.clean_exit else "TASK MANAGER error exit"
    #         if self.user_config["send_emails"]:
    #             send_email(subject, "no msg", receiver_email=self.user_config["email"])
