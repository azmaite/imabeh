""" Test functions for imabeh tasks:
    - main.get_dx
    - main.find_steps
"""
# fictrac test on dataset takes ~30 sec
# df3d test on dataset takes ~2 min 30 sec

import unittest
import os

# Copy the current user_and_dirs_to_process.txt file to revert to later
# Change the user to the test user so that all the imports happen correctly
LOCAL_DIR, _ = os.path.split(os.path.realpath(__file__))
LOCAL_DIR = LOCAL_DIR.rstrip("tests") + "run"
txt_file = os.path.join(LOCAL_DIR, "_user_and_fly_dirs_to_process.txt")
txt_file_base, txt_file_ext = os.path.splitext(txt_file)
txt_file_copy = txt_file_base + "_copy" + txt_file_ext
os.system(f"cp {txt_file} {txt_file_copy}")

# write the test user
with open(txt_file, "w") as f:
    f.write("CURRENT_USER = USER_TEST")
    f.write("\n")

from imabeh.run.userpaths import GLOBAL_PATHS, LOCAL_DIR

from imabeh.tests.main import replace_txt_file, restore_txt_file, delete_log
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

    @classmethod
    def tearDownClass(cls):
        # This will run after all tests in this class have finished

        # Delete the _fly_processing_status files
        from imabeh.run.userpaths import get_current_user_config
        user_config = get_current_user_config()
        fly_status_file = os.path.join(user_config['labserver_data'], user_config["csv_fly_table"])
        os.system(f"rm {fly_status_file}")
        fly_status_base, fly_status_ext = os.path.splitext(fly_status_file)
        fly_status_file_copy = fly_status_base + "_backup" + fly_status_ext  # This ensures the extension is preserved
        print('HERE')
        print(fly_status_file_copy)
        os.system(f"rm {fly_status_file_copy}")

        # Restore the original txt file
        restore_txt_file()

    def test_user(self):
        # Test whether the user is read correctly and it can find the correct dataset

        test_line = test_data_dir + "/Fly1||001_beh-stack||df"
        replace_txt_file(test_line)

        # check the user
        try:
            from imabeh.run.userpaths import get_current_user_config
            user_config = get_current_user_config()
        except:
            self.fail("Failed to read the user from the txt file - maybe USER_TEST missing from userpaths?.")

        self.assertEqual(user_config['initials'], 'TEST', "Wrong user read from the txt file.")


    def test_fly_dir_error(self):
        # Test whether script removes the fly if the fly directory does not exist

        test_line = test_data_dir + "/Fly___||001_beh-stack||df"

        replace_txt_file(test_line)
        log = LogManager(log_name="test_log")
        torun_table = TaskManager(log).torun_table
        delete_log(log)

        # check that the torun table has 0 rows
        rows = len(torun_table)
        self.assertEqual(rows, 0, "Failed to detect inexistent fly dirs.")


    def test_task_error(self):
        # Test whether script removes the task if it is not included in the task_collection
        
        test_line = test_data_dir + "/Fly1||001_beh-stack||___"
        log = LogManager(log_name="test_log")

        replace_txt_file(test_line)
        torun_table = TaskManager(log).torun_table
        delete_log(log)

        # check that the torun table has 0 rows
        rows = len(torun_table)
        self.assertEqual(rows, 0, "Failed to detect inexistent tasks.")

    
    def test_complete_trials(self):        
        # Test whether script completes trial names

        test_line = test_data_dir + "/Fly1||001||df"
        trials_expected = ['001_beh-stack']

        replace_txt_file(test_line)
        log = LogManager(log_name="test_log")
        trials = TaskManager(log).torun_table.trial.values.tolist()
        delete_log(log)
        print(trials, trials_expected)

        self.assertListEqual(trials, trials_expected, "Failed to complete partially provided trials.")


    def test_get_all_trials(self):
        # Test whether script gets all trials when requested

        test_line = test_data_dir + "/Fly1||all||df"
        trials_expected = ['001_beh-stack','002_beh-singleZ','003_beh-only']
        trials_expected.sort() # sort to compare
        
        replace_txt_file(test_line)
        log = LogManager(log_name="test_log")
        trials = TaskManager(log).torun_table.trial.values.tolist()
        trials.sort()
        delete_log(log)

        self.assertListEqual(trials, trials_expected, "Failed to get all trials.")


    def test_keywords(self):
        # Test whether script gets trials based on keywords

        test_line = test_data_dir + "/Fly1||k-stack||df"
        trials_expected = ['001_beh-stack']

        replace_txt_file(test_line)
        log = LogManager(log_name="test_log")
        trials = TaskManager(log).torun_table.trial.values.tolist()
        delete_log(log)

        self.assertListEqual(trials, trials_expected, "Failed to get trials based on keyword.")

    
    def test_exclude_trials(self):
        # Test whether script excludes trials

        test_line = test_data_dir + "/Fly1||all,e-001||df"
        trials_expected = ['002_beh-singleZ','003_beh-only']
        trials_expected.sort() # sort to compare

        replace_txt_file(test_line)
        log = LogManager(log_name="test_log")
        trials = TaskManager(log).torun_table.trial.values.tolist()
        trials.sort()
        delete_log(log)

        self.assertListEqual(trials, trials_expected, "Failed exclude trials (will also fail if get_all_trials and complete_trials fail).")


    def test_pipelines(self):
        # Test whether script correctly replaces pipelines with tasks

        test_line = test_data_dir + "/Fly1||001_beh-stack||p-test"
        tasks_expected = ['tif','df']
        tasks_expected.sort() # sort to compare

        replace_txt_file(test_line)
        log = LogManager(log_name="test_log")
        tasks = TaskManager(log).torun_table.task.values.tolist()
        tasks.sort()
        delete_log(log)

        print(tasks)
        print(tasks_expected)
        
        self.assertListEqual(tasks, tasks_expected, "Failed exclude trials (will also fail if get_all_trials and complete_trials fail).")

        




# class TestTasks(unittest.TestCase):
#     """ 
#     Tests the tasks available in imabeh script:
#         - df3d
#         - fictrac
#         - tif
#         - flat
#         - df
#     """
#     txt_file = GLOBAL_PATHS["txt_user_and_dirs_to_process"]

#     def test_df3d(self, txt_file=txt_file):
#         # test whether df3d runs correctly

#         test_line = test_data_dir + "||001||df3d"
#         _replace_txt_file(test_line)

#         # check that df3d runs without errors
#         try:
#             run.main()
#         except:
#             self.fail("Failed while running df3d.")

#         # check that df3d output is correct (all files present)
        

#         _delete_all_created_files(test_data_dir)
    

    

if __name__ == '__main__':
    # will stop if a test fails
    unittest.main()
