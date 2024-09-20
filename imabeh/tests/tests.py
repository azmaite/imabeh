""" Test functions for imabeh tasks:
    - main.get_dx
    - main.find_steps
"""
# fictrac test on dataset takes ~30 sec
# df3d test on dataset takes ~2 min 30 sec

import unittest
import os
import json

from imabeh.run.userpaths import GLOBAL_PATHS, LOCAL_DIR

from imabeh.run.logmanager import LogManager
from imabeh.run.taskmanager import TaskManager
from imabeh.run import run

test_dir, _ = os.path.split(os.path.realpath(__file__))

test_data_dir = '240912_test-dataset'


class TestReadTxtFile(unittest.TestCase):
    """ Test the _read_txt_file function in taskmanager.py.
    Checks that it reads the user and directories to process correctly.
    """
    txt_file = GLOBAL_PATHS["txt_user_and_dirs_to_process"]

    def test_user(self):
        # Test whether the user is read correctly and it can find the correct dataset

        test_line = test_data_dir + "Fly1||001_beh-stack||df"
        _replace_txt_file(test_line)

        # check the user
        try:
            from imabeh.run.userpaths import user_config
        except:
            self.fail("Failed to read the user from the txt file - maybe USER_TEST missing from userpaths?.")
        self.assertEqual(user_config['initials'], 'TEST', "Wrong user read from the txt file.")

        _restore_txt_file_delete_log()

    def test_fly_dir_error(self):
        # Test whether script removes the fly if the fly directory does not exist

        test_line = test_data_dir + "Fly___||001_beh-stack||df"

        _replace_txt_file(test_line)
        log = LogManager(log_name="test_log")
        torun_table = TaskManager(log).torun_table

        # check that the torun table has 0 rows
        rows = len(torun_table)
        self.assertEqual(rows, 0, "Failed to detect inexistent fly dirs.")

        _restore_txt_file_delete_log(log)


    def test_task_error(self):
        # Test whether script removes the task if it is not included in the task_collection
        
        test_line = test_data_dir + "Fly1||001_beh-stack||___"
        log = LogManager(log_name="test_log")

        _replace_txt_file(test_line)
        torun_table = TaskManager(log).torun_table

        # check that the torun table has 0 rows
        rows = len(torun_table)
        self.assertEqual(rows, 0, "Failed to detect inexistent tasks.")

        _restore_txt_file_delete_log(log)

    
    def test_complete_trials(self):        
        # Test whether script completes trial names

        test_line = test_data_dir + "Fly1||001||df"
        trials_expected = ['001_beh-stack']

        _replace_txt_file(test_line)
        log = LogManager(log_name="test_log")
        trials = TaskManager(log).torun_table.trial.values

        self.assertListEqual(trials, trials_expected, "Failed to complete partially provided trials.")

        _restore_txt_file_delete_log(log)


    def test_get_all_trials(self):
        # Test whether script gets all trials when requested

        test_line = test_data_dir + "Fly1||all||df"
        trials_expected = ['001_beh-stack','002_beh-singleZ','003_beh-only']
        
        _replace_txt_file(test_line)
        log = LogManager(log_name="test_log")
        trials = TaskManager(log).torun_table.trial.values

        self.assertListEqual(trials, trials_expected, "Failed to get all trials.")

        _restore_txt_file_delete_log(log)


    def test_keywords(self):
        # Test whether script gets trials based on keywords

        test_line = test_data_dir + "Fly1||k-stack||df"
        trials_expected = ['001_beh-stack']

        _replace_txt_file(test_line)
        log = LogManager(log_name="test_log")
        trials = TaskManager(log).torun_table.trial.values

        self.assertListEqual(trials, trials_expected, "Failed to get trials based on keyword - found too .")

        _restore_txt_file_delete_log(log)

    
    def test_exclude_trials(self):
        # Test whether script excludes trials

        test_line = test_data_dir + "Fly1||all,e-001||df"
        trials_expected = ['002_beh-singleZ','003_beh-only']

        _replace_txt_file(test_line)
        log = LogManager(log_name="test_log")
        trials = TaskManager(log).torun_table.trial.values
        self.assertListEqual(trials, trials_expected, "Failed exclude trials (will also fail if get_all_trials and complete_trials fail).")

        _restore_txt_file_delete_log(log)


    def test_pipelines(self):
        # Test whether script correctly replaces pipelines with tasks

        test_line = test_data_dir + "Fly1||001_beh-stack||test"
        tasks_expected = ['tif','df']

        _replace_txt_file(test_line)
        log = LogManager(log_name="test_log")
        tasks = TaskManager(log).torun_table.task.values
        self.assertListEqual(tasks, tasks_expected, "Failed exclude trials (will also fail if get_all_trials and complete_trials fail).")

        _restore_txt_file_delete_log(log)




class TestTasks(unittest.TestCase):
    """ 
    Tests the tasks available in imabeh script:
        - df3d
        - fictrac
        - tif
        - flat
        - df
    """
    txt_file = GLOBAL_PATHS["txt_user_and_dirs_to_process"]

    def test_df3d(self, txt_file=txt_file):
        # test whether df3d runs correctly

        test_line = test_data_dir + "||001||df3d"
        _replace_txt_file(test_line)

        # check that df3d runs without errors
        try:
            run.main()
        except:
            self.fail("Failed while running df3d.")

        # check that df3d output is correct (all files present)
        

        _restore_txt_file_delete_log()
        _delete_all_created_files(test_data_dir)
    


## ACCESSORY FUNCTIONS

def _replace_txt_file(line):
    """ Replace the txt_file with the flies/tasks provided. 
    Parameters
    ----------
    line : str
        The line to write to the txt file (which flies/tasks to run for testing)
   """

    # make a copy of the current user_and_dirs_to_process.txt (ro revert to later)
    txt_file = os.path.join(LOCAL_DIR, GLOBAL_PATHS["txt_user_and_dirs_to_process"])
    txt_file_copy = txt_file.rstrip(".txt") + "_copy.txt"
    os.system(f"cp {txt_file} {txt_file_copy}")

    # write the test user and test flies/tasks to the txt file
    with open(txt_file, "w") as f:
        f.write("CURRENT_USER = USER_TEST")
        f.write("\n")
        f.write(line)

def _restore_txt_file_delete_log(log = None):
    """ Restore the txt_file to its original state. 
    If log is provided, it also deletes the log file."""

    txt_file = os.path.join(GLOBAL_PATHS["txt_user_and_dirs_to_process"])
    txt_file_copy = txt_file.rstrip(".txt") + "_copy.txt"

    os.system(f"rm {txt_file}") 
    os.system(f"mv {txt_file_copy} {txt_file}")

    if log is not None:
        os.system(f"rm {os.path.join(LOCAL_DIR, 'logs', log.log_file)}")

def _record_current_state(test_data_dir=test_data_dir, recorded_structure_file="test_data_files.json"):
    """ Record the current state of the directory structure to a JSON file.
    Used to generate the record of original files so any new ones
    created by testing can be deleted.
    SHOULDN'T NEED TO EVER BE USED AGAIN!!"""

    test_data_dir = os.path.join(LOCAL_DIR.rstrip('run'),'tests', test_data_dir)

    directory_structure = []

    # Walk through the directory and collect all file and folder paths
    for dirpath, dirnames, filenames in os.walk(test_data_dir):
        for dirname in dirnames:
            directory_structure.append(os.path.join(dirpath, dirname))
        for filename in filenames:
            directory_structure.append(os.path.join(dirpath, filename))

    # Save the directory structure to a JSON file
    output_path = os.path.join(os.path.dirname(test_data_dir), recorded_structure_file)
    with open(output_path, 'w') as f:
        json.dump(directory_structure, f, indent=4)

def _delete_all_created_files(test_data_dir, recorded_structure_file="test_data_files.json"):
    """ Delete all files created by the tests within the testdata folder.
    Check the original directory structure and delete any new files/folders."""

    test_data_dir = os.path.join(LOCAL_DIR.rstrip('run'),'tests', test_data_dir)
    file_list_path = os.path.join(os.path.dirname(test_data_dir), recorded_structure_file)

    # Load the recorded structure
    with open(file_list_path, 'r') as f:
        recorded_structure = set(json.load(f))

    # Get the current state of the directory
    current_structure = set()
    for dirpath, dirnames, filenames in os.walk(test_data_dir):
        for dirname in dirnames:
            current_structure.add(os.path.join(dirpath, dirname))
        for filename in filenames:
            current_structure.add(os.path.join(dirpath, filename))

    # Find new files/folders
    new_entries = current_structure - recorded_structure

    # Delete new files/folders
    for entry in new_entries:
        if os.path.isfile(entry):
            os.remove(entry)
        elif os.path.isdir(entry):
            os.rmdir(entry)
    

if __name__ == '__main__':
    unittest.main()
