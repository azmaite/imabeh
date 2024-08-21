""" Test functions:
    - runutils.read_current_user
"""

import unittest
import os

from imabeh.run.runutils import read_current_user, read_fly_dirs


class TestGetCurrentUser(unittest.TestCase):
    """ Test the runutils.read_current_user function. """

    def test_error_no_file(self):
        """ _Test that the function raises a FileNotFoundError if the file does not exist. """
        with self.assertRaises(FileNotFoundError):
            read_current_user("nonexistent.txt")

    def test_error_no_current_user(self):  
        """ _Test that the function raises a ValueError if the file does not contain the CURRENT_USER variable. """
        with open("test.txt", "w") as file:
            file.write("USER_XXX")
        with self.assertRaises(ValueError):
            read_current_user("test.txt")
        os.remove("test.txt")

    def test_error_no_userpaths(self):      
        """ _Test that the function raises a ValueError if the user does not exist in the userpaths file. """
        with open("test.txt", "w") as file:
            file.write("CURRENT_USER = USER_XXX")
        with self.assertRaises(ValueError):
            read_current_user("test.txt")
        os.remove("test.txt")

class TestGetFlyDirs(unittest.TestCase):
    """ Test the runutils.read_current_user function. """

    def test_error_no_file(self):
        """ _Test that the function raises a FileNotFoundError if the file does not exist. """
        with self.assertRaises(FileNotFoundError):
            read_fly_dirs("nonexistent.txt")

    def test_error_wrong_format_short(self):  
        """ _Test that the function raises a ValueError if the file list format is too short. """
        with open("test.txt", "w") as file:
            file.write("Fly_dir||Trial")
        with self.assertRaises(ValueError):
            read_fly_dirs("test.txt")
        os.remove("test.txt")
    
    def test_error_wrong_format_long(self):  
        """ _Test that the function raises a ValueError if the file list format is too long. """
        with open("test.txt", "w") as file:
            file.write("Fly_dir||Trial||task1||extra_input")
        with self.assertRaises(ValueError):
            read_fly_dirs("test.txt")
        os.remove("test.txt")

    def test_error_no_fly_file(self):
        """ _Test that the function raises a FileNotFoundError if a fly directory does not exist. """
        with open("test.txt", "w") as file:
            file.write("Fake_file||Fake_trial||task1")
        with self.assertRaises(FileNotFoundError):
            read_fly_dirs("test.txt")
        os.remove("test.txt")



if __name__ == '__main__':
    unittest.main()
