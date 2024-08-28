# imabeh
Processing pipeline for Drosophila imaging and behavior data 
Many things copied or inspired from twoppp



Outline of running process:

1. run the run.py script, which:
    - creates a log :                   LogManager.create_task_log()
    -* initializes the TaskManger :     TaskManager()
    - logs the start of processing :    LogManager.add_line_to_log()
    -* runs the TaskManager :           TaskManager.run()
2. initalize the TaskManager, which:
    - sets the base properties
    - gets the list of trials/tasks to process from the txt file, as trial_dicts :      TaskManager._read_fly_dirs()
    - gets the torun_table, icnluding all tasks that must be run :                       TaskManager._create_torun_table()
      . adds each trial+task (a torun) to the torun_list, and whether to overwrite (!) :  TaskManager._add_toruns_from_trial()
      . checks for duplicate toruns and removes if so (+log) :                           TaskManager._check_duplicates()   
      . checks if any of the toruns have already been completed using the FlyTable :     FlyTableManager.check_trial_task_status()
      . removes any toruns that have been done and should not be overwritten (+log) :    TaskManager._remove_torun()
      . checks the pre-requisites for each task and sets statuses acordingly :          TaskManager._check_prerequisites()        
      . concludes with sorted to-do list!
3. run the TaskManager, which: 
    - 

