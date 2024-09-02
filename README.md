# imabeh
Processing pipeline for Drosophila imaging and behavior data 
Many things copied or inspired from twoppp



Outline of running process:

1. run the run.py script, which:
    - creates a log :                       LogManager.create_task_log()
    - (2.) initializes the TaskManger :     TaskManager()
    - logs the start of processing :        LogManager.add_line_to_log()
    - (3.) runs the TaskManager :           TaskManager.run()

2. initalize the TaskManager, which:
    - sets the base properties
    - gets the list of trials/tasks to process from the txt file, as trial_dicts :       TaskManager._read_fly_dirs()
    - gets the torun_table, icnluding all tasks that must be run :                       TaskManager._create_torun_table()
      . adds each trial+task (a torun) to the torun_list, and whether to overwrite (!) : TaskManager._add_toruns_from_trial()
      . checks for duplicate toruns and removes if so (+log) :                           TaskManager._check_duplicates()   
      . checks that all tasks exist in the task_collection :                             TaskManager._check_task_exists()
      . checks if any of the toruns have already been completed using the FlyTable :     FlyTableManager.check_trial_task_status()
      . removes any toruns that have been done and should not be overwritten (+log) :    TaskManager._remove_torun()
      . checks the pre-requisites for each task and sets statuses acordingly :           TaskManager._check_prerequisites()        
      . CONCLUDES WITH A SORTED TO-RUN LIST!

3. run the TaskManager, which recursively: 
    - checks if there are any tasks left to run, and exits if not 
    - checks all "running" tasks to see if they finished :                               TaskManager._check_task_finished()
        - if any finished correctly or failed, remove from table (+ update fyTable)
    - if any status changed, check all prereqs and update status/remove todos (+Log) :   TaskManager._check_prerequisites()
    - checks if any tasks are "ready"
        - if all tasks are "waiting"/"running", wait for self.t_wait_s before restart                        
    - selects and runs the next task, changing its status to "running" :                 TaskManager._execute_next_task()
    
To check whether any task has finished, regardless of if it is run on python or otherwise, it will check for the existence of a
task status log file and its contents (see imabeh.run.tasks, start_run and test_finished).
Make sure any non-python processes generate this file correctly when done!
