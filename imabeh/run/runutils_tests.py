""" Test functions:
    - runutils.read_current_user
"""

import unittest
import os

from imabeh.run.runutils import read_current_user


class TestGetCurrentUser(unittest.TestCase):
    """ Test the runutils.read_current_user function. """

    def test_error_no_file(self):
        """ Test that the function raises a FileNotFoundError if the file does not exist. """
        with self.assertRaises(FileNotFoundError):
            read_current_user("nonexistent.txt")

    def test_error_no_current_user(self):  
        """ Test that the function raises a ValueError if the file does not contain the CURRENT_USER variable. """
        with open("test.txt", "w") as file:
            file.write("USER_XXX")
        with self.assertRaises(ValueError):
            read_current_user("test.txt")
        os.remove("test.txt")

    def test_error_no_userpaths(self):      
        """ Test that the function raises a ValueError if the user does not exist in the userpaths file. """
        with open("test.txt", "w") as file:
            file.write("CURRENT_USER = USER_XXX")
        with self.assertRaises(ValueError):
            read_current_user("test.txt")
        os.remove("test.txt")

if __name__ == '__main__':
    unittest.main()
