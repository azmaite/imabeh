#!/usr/bin/env python3
"""
run script to run processing with the TaskManager class.
See the README.md file in the same folder for usage instructions
"""

################
from imabeh.run.runparams import global_params ######### FIX

from imabeh.run.tasks import task_collection
from imabeh.run.taskmanager import TaskManager
from imabeh.run.logmanager import create_task_log, add_line_to_log
from imabeh.run.userpaths import get_current_user_config

# get current user settings and paths
user_config = get_current_user_config()

def main() -> None:
    """
    main function to initialise and run processing
    """

    # create log
    log = create_task_log()
    
    # initialize task manager
    task_manager = TaskManager(task_collection, params=global_params, user_config=user_config, log)

    # log start of processing and list of tasks to do
    log.add_line_to_log("START: Will start task manager with the following tasks:")
    for todo_dict in task_manager.todo_dicts:
        log.add_line_to_log(f"   {todo_dict['tasks']}: {todo_dict['dir']}")
    log.add_line_to_log("\n")

    # run task manager
    task_manager.run()

if __name__ == "__main__":
    main()
