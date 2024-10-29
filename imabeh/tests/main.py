import os
import json

from imabeh.run.userpaths import GLOBAL_PATHS, LOCAL_DIR

def replace_txt_file(line):
    """ Replace the txt_file with the flies/tasks provided. 
    Parameters
    ----------
    line : str
        The line to write to the txt file (which flies/tasks to run for testing)
   """

    txt_file = os.path.join(LOCAL_DIR, GLOBAL_PATHS["txt_user_and_dirs_to_process"])
    with open(txt_file, "w") as f:
        f.write("CURRENT_USER = USER_TEST")
        f.write("\n")
        f.write(line)

def restore_txt_file():
    """ Restore the txt_file to its original state."""

    txt_file = os.path.join(GLOBAL_PATHS["txt_user_and_dirs_to_process"])
    txt_file_base, txt_file_ext = os.path.splitext(txt_file)
    txt_file_copy = txt_file_base + "_copy" + txt_file_ext

    os.system(f"rm {txt_file}") 
    os.system(f"mv {txt_file_copy} {txt_file}")

def delete_log(log):
    """ Deletes the log file."""
    os.system(f"rm {os.path.join(LOCAL_DIR, 'logs', log.log_file)}")





def record_current_state(test_data_dir, recorded_structure_file="test_data_files.json"):
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

def delete_all_created_files(test_data_dir, recorded_structure_file="test_data_files.json"):
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