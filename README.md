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
    - gets the todo_table, icnluding all tasks that must be run :                       TaskManager._create_todo_table()
      . adds each trial+task (a todo) to the todo_list, and whether to overwrite (!) :  TaskManager._add_todos_from_trial()
      . checks if any of the todos have already been completed using the FlyTable :     FlyTableManager.check_trial_task_status()
      . removes any todos that have been done and should not be overwritten (+log) :    TaskManager._remove_todo()
      . checks the pre-requisites for each task and sets statuses acordingly :          TaskManager._check_prerequisites()       
      . concludes with sorted to-do list!
3. run the TaskManager, which: 
    - 

