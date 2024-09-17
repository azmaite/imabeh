#!/usr/bin/env python3
"""
run script to run processing with the TaskManager class.
See the README.md file in the same folder for usage instructions
"""

################
#from imabeh.run.runparams import global_params ######### FIX

from imabeh.run.tasks import task_collection
from imabeh.run.taskmanager import TaskManager
from imabeh.run.logmanager import LogManager
from imabeh.run.userpaths import user_config # get the current user configuration (paths and settings)


def main() -> None:
    """
    main function to initialise and run processing
    """

    # create log
    log = LogManager()
    
    # initialize task manager
    task_manager = TaskManager(log, user_config, task_collection) ######### FIX params=global_params, 

    # log start of processing and list of tasks to do
    log.add_line_to_log("Will start task manager with the following tasks:")
    for _, torun in task_manager.torun_table.iterrows():
        log.add_line_to_log("%-*s - %s" % (40, f"{torun.fly_dir} / {torun.trial}",torun.task))

    # run task manager
    task_manager.run(log)

if __name__ == "__main__":
    main()
