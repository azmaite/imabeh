"""
task manager module housing the TaskManager class
"""
from copy import deepcopy
from typing import List, Dict
import time
import numpy as np
import datetime
import pandas as pd

#from twoppp.pipeline import PreProcessParams

#from twoppp.run.runutils import read_fly_dirs, read_running_tasks, check_task_running
#from twoppp.run.runutils import send_email, write_running_tasks
from imabeh.run.tasks_2 import Task

from imabeh.run.logmanager import LogManager, add_line_to_log
from imabeh.run.userpaths import LOCAL_DIR, read_current_user
from imabeh.run.flytablemanager import check_trial_task_status

# get the current user configuration (paths and settings)
user_config = read_current_user()


class TaskManager():
    """
    class to prioritise and sequentially execute different tasks
    """
    def __init__(self, 
                 task_collection: Dict[str, Task],
                 params: PreProcessParams, 
                 user_config: dict=user_config,
                 log: LogManager
            ) -> None:
        """
        Most important component is a list of tasks todo that is managed: self.todo_dicts
        A "todo" ITEM IS DEFINED AS A SINGLE TASK TO BE RUN ON A SINGLE TRIAL
        Each todo item will have the following fields:
            - idx: the index of the todo in the order of tasks to run 
            - dir: the base directory of the fly, where the data is stored
            - trial: the fly trial to analyze
            - tasks: the name of the task to be done. Later used to index in self.task_collection
            - overwrite: whether or not to force an overwrite of the previous results
            - status: whether the to_run is "ready", "running", "done", or "waiting"

        Parameters
        ----------
        task_collection : Dict[str, Task]
            a dictionary indexed by strings containing instances of the class Task.
            This dictionary should contain all possible tasks
        params : PreProcessParams
            parameters for pre-processing. #############################################
        user_config : dict
            dictionary with user specific parameters, such as file directories, etc.
        log : LogManager
        """

        # set base properties
        self.task_collection = task_collection
        self.params = deepcopy(params)
        self.user_config = user_config
        self.txt_file_to_process = self.user_config["txt_file_to_process"]
        self.txt_file_running = self.user_config["txt_file_running"]
        # set time to wait between recurrently checking for non-python tasks to run
        self.t_wait_s = 60

        #####
        self.clean_exit = False

        # get the list of fly trials and tasks from the "txt_file_to_process"
        # one dict per trial, with many tasks possible in each
        self.trials_to_process = []
        self._read_fly_dirs(self.txt_file_to_process)

        # get table of flies/tasks to run (todos) from trials_to_process
        # same as previous, but separate tasks (one row each)
        # Also check if should overwrite and add order based on txt file
        # first create empty table
        header = ["fly_dir", "trial", "task", "overwrite", "status"]
        self.todo_table = pd.DataFrame(columns=header)
        # fill with new todos
        for trial_dict in self.trials_to_process:
            self._add_todos_from_trial(self, trial_dict)

        # check if any tasks have been completed previously (without errors) using the fly table
        # and remove todo's if so
        for index, todo in self.todo_table.iterrows():
            # convert todo to dict
            todo_dict = todo.to_dict()
            # check status in table
            status = check_trial_task_status(todo_dict)
            # if already completed and overwrite = False, log and remove from list of to_run
            if status == 1 and todo.overwrite == False: # 1 = done
                self._remove_todo()
                

        





    @property
    def status(self) -> List[str]:
        """
        get the current status of each of the todos registered by accessing the status of each dict.
        Either "running", "done", "waiting", "ready"

        Returns
        -------
        List[str]
        """
        return [todo["status"] for todo in self.todo_dicts]

    @property
    def prios(self) -> np.ndarray:
        """
        get the priority levels of each of the todos registered by accesing the prio of the Task
        associated with each todo.

        Returns
        -------
        np.ndarray
        """
        return np.array([self.task_collection[todo["tasks"]].prio for todo in self.todo_dicts])

    @property
    def n_todos(self) -> int:
        """
        get the number of todo items.

        Returns
        -------
        int
        """
        return len(self.todo_dicts)



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
            - "dir": the base directory of the fly
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
            # split into sections (dir,trials,tasks)
            strings = line.split("||")
            # split trials and make a dict for each
            trials = strings[1]
            for trial in trials:
                trial_dict = {
                    "dir": strings[0],
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
            - "dir": the base directory of the fly
            - "trials": which trials to run on
            - "tasks": a comma separated string containing the names of the tasks to run
        """

        # split tasks
        task_list = trial_dict["tasks"].split(',')

        # get last todo order in table
        order = self.todo_table.index.max()

        # for each task, make a new todo dict (will become a row)
        for task_name in task_list:
            new_todo = deepcopy(trial_dict)

            # check if tasks must be overwritten
            if task_name.startswith("!"):
                new_todo["tasks"] = task_name[1:]
                new_todo["overwrite"] = True
            else:
                new_todo["tasks"] = task_name
                new_todo["overwrite"] = False

            # set status to ready (default)
            new_todo["status"] = "ready"
           
            # add todo to table as new row
            order += order
            self.todo_table.loc[order] = (new_todo)
        
        # sort indexes (order)
        self.todo_table = self.todo_table.sort_index()


    def _remove_todo(self, trial_dict : dict):
        """
        function to remove a todo row from the self.todo_table

        Parameters
        ----------
        trial_dict: dict
            trial dict with the following fields for each fly trial:
            - "dir": the base directory of the fly
            - "trial": which trial to run on
            - "tasks": a comma separated string containing the names of the tasks to run
        """

        # split tasks
        task_list = trial_dict["tasks"].split(',')

        # get last todo order in table
        order = self.todo_table.index.max()

        # for each task, make a new todo dict (will become a row)
        for task_name in task_list:
            new_todo = deepcopy(trial_dict)

            # check if tasks must be overwritten
            if task_name.startswith("!"):
                new_todo["tasks"] = task_name[1:]
                new_todo["overwrite"] = True
            else:
                new_todo["tasks"] = task_name
                new_todo["overwrite"] = False

            # set status to ready (default)
            new_todo["status"] = "ready"
           
            # add todo to table as new row
            order += order
            self.todo_table.loc[order] = (new_todo)
        
        # sort indexes (order)
        self.todo_table = self.todo_table.sort_index()






## renamed
    def check_task_status(self, fly_dict: dict, running_tasks: List[dict]) -> List[dict]:
        """
        check which tasks to do for a given fly by calling the test_todo() method of each task
        and checking whether a task is already running.

        Parameters
        ----------
        fly_dict : dict
            dictionary with info about the fly to process. Has to contain the following fields:
            - dir: the base directory, where the data is stored
            - tasks: a string describing which tasks todo, e.g.: "pre_cluster,post_cluster"
            - selected_trials: a string describing which trials to run on,
                               e.g. "001,002" or "all_trials"
        running_tasks : List[dict]
            a list of tasks that is currently running, e.g., as obtained by read_running_tasks().
            Essentially a list of dictionaries like the "fly_dict" specified above.

        Returns
        -------
        List[dict]
            a list of fly_dicts that have processing yet to be done.
        """
        task_names = fly_dict["tasks"].split(",")
        todos = []
        for task_name in task_names:
            if task_name.startswith("!"):
                task_name_orig = task_name
                task_name = task_name[1:]
                force = True
            else:
                force = False
                task_name_orig = task_name
            if task_name not in self.task_collection.keys():
                print(f"{task_name} is not an available pre-processing step.\n" +
                    f"Available tasks: {self.task_collection.keys()}")
            elif check_task_running(fly_dict, task_name, running_tasks) and \
                self.user_config["check_tasks_running"]:
                print(f"{task_name} is already running for fly {fly_dict['dir']}.")
            else:
                if force or self.task_collection[task_name].test_todo(fly_dict):
                    todos.append(task_name_orig)
        return todos



    def remove_to_run(self, i_todo: int) -> None:
        """
        remove a todo item from the self.todo_dicts list and from the running_tasks text file.

        Parameters
        ----------
        i_todo : int
            index of the todo in the self.todo_dicts list
        """
        todo = self.todo_dicts.pop(i_todo)
        write_running_tasks(todo, add=False)

    def log_failed_task(self, task_name: str, exception: Exception) -> None:
        """
        Log the failed task to a text file with the date and time.

        Parameters
        ----------
        task_name : str
            The name of the task that failed.
        exception : Exception
            The exception that was raised.
        """
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open("failed_tasks.txt", "a") as f:
            f.write(f"[{current_time}] Task {task_name} failed with exception: {exception}\n")





    def execute_next_task(self) -> bool:
        """
        Execute the next todo from self.todo_dicts that is not waiting.

        Returns
        -------
        bool
            True if a task was run to completion, False if all tasks are "waiting.
        """

        result = False

        for i_todo, next_todo in enumerate(self.todo_dicts):
            task_name = next_todo["tasks"]
            write_running_tasks(next_todo, add=True)
            self.todo_dicts[i_todo]["status"] = "running"
            try:
                result = self.task_collection[task_name].run(fly_dict=next_todo, params=self.params)
                if result:
                    self.todo_dicts[i_todo]["status"] = "done"
                    self.remove_todo(i_todo)
                    return True
                else:
                    self.todo_dicts[i_todo]["status"] = "waiting"
                    write_running_tasks(next_todo, add=False)
            except Exception as e:
                # Log the exception and continue with the next task
                print(f"Task {task_name} failed with exception: {e}")
                self.todo_dicts[i_todo]["status"] = "failed"
                write_running_tasks(next_todo, add=False)
                self.log_failed_task(task_name, e)
        return False







    def run(self) -> None:
        """
        run the tasks manager to sequentially process all todos from self.todo_dicts.
        """
        try:
            # check if there are tasks left to run
            while self.n_todos:
                # run the next task and check if it works
                success = self.execute_next_task()


                if not success:
                    print('Waiting because of error - Check that sync file exists in folder.')
                    # if all tasks are pending wait before checking from start again
                    time.sleep(self.t_wait_s)

            self.clean_exit = True

        finally:
            print("TASK MANAGER clean up: removing tasks from _tasks_running.txt")
            for todo in self.todo_dicts:
                if todo["status"] == "running":
                    write_running_tasks(todo, add=False)
            subject = "TASK MANAGER clean exit" if self.clean_exit else "TASK MANAGER error exit"
            if self.user_config["send_emails"]:
                send_email(subject, "no msg", receiver_email=self.user_config["email"])
