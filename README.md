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
    - adds each trial+task (a todo) to the todo_list, and whether to overwrite (!) :    TaskManager._add_todos_from_trial()
    - checks if any of the todos have already been done :                               FlyTableManager.check_trial_task_status()
    - removes any todos that have been done and should not be overwritten (+log) :      ......
    - optionally reorders trial+tasks :                                                 ......
    - checks the pre-requisites for each task and sets statuses acordingly :            ......         
    - concludes with sorted to-do list!

