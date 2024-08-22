"""
task manager module housing the TaskManager class
"""
from copy import deepcopy
from typing import List, Dict
import time
import numpy as np
import datetime

from imabeh.run.userpaths import GLOBAL_PATHS
from imabeh.imabeh.run.new_versions_not_used.runutils import read_current_user, read_fly_dirs
from imabeh.imabeh.run.new_versions_not_used.runutils import get_fly_table, update_fly_table, save_fly_table
from imabeh.imabeh.run.new_versions_not_used.runutils import create_task_log, add_line_to_log

from imabeh.imabeh.run.new_versions_not_used.tasks import task_collection


class TaskManager():
    """
    class to sequentially execute different tasks according to _fly_dirs_to_process.txt file.
    """
    def __init__(self) -> None:
        """
        class to sequentially execute different tasks on different flies and trials.
        Most important component is a list of todos that is managed: self.todo_dicts
        Each todo item will have the following fields:
            - dir: the base directory of the fly, where the data is stored
            - selected_trials: a string describing which trials to run on,
                               e.g. "001,002" or "all_trials"
            - overwrite: whether or not to force an overwrite of the previous results
            - tasks: the name of the task to be done. Later used to index in self.task_collection
            - status: whether the todo is "ready","running", "done", or "waiting"

        """


    ##############################################
        
    def execute_next_task(self) -> bool:
        """
        execute the next task in the todo list.

        Returns
        -------
        bool
            True if a task was run successfuly, False if not.
        """

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

        # create a new log file
        create_task_log()

        # get list of tasks to do
        todo_dict = self.get_tasks_todo()

        # loop over all tasks to do
        for todo in todo_dict:
            # check if task alraedy completed
            # if yes, log
            #   if overwrite - continue
            #   else - skip



            # log start of task
            add_line_to_log(f"Starting task {todo['tasks']} on fly {todo['dir']}")
            # run the task
            self.execute_next_task(todo)
            # check if finished and if successful or not
            #if todo["status"] == "done":


            # log end of task
            add_line_to_log(f"Finished task {todo['tasks']} on fly {todo['dir']}")
            # update the fly table
            update_fly_table(todo['dir'], todo['tasks'])

        # save the fly table
        save_fly_table()





        """
        find out for which flies processing has to be run by reading the _fly_dirs_to_process file.
        """
        fly_dicts = read_fly_dirs(GLOBAL_PATHS["txt_file_to_process"])


        running_tasks = read_running_tasks(self.txt_file_running)
        for fly_dict in fly_dicts:
            fly_dict["todos"] = self.check_which_tasks_todo(fly_dict, running_tasks)
            if len(fly_dict["todos"]) == 0:
                continue
            else:
                self.flies_to_process.append(fly_dict)
        print("Will run processing on the following flies: " +\
              f"\n{[fly['dir'] for fly in self.flies_to_process]}")