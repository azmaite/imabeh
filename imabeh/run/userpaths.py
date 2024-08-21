"""
sub-module to define paths and user settings for running pre-processing with.

to adapt the parameters for your use:
1. create a dictionary for your user and define the paths and parameters accordingly
2. set CURRENT_USER at the top of the "_fly_dirs_to_process.txt" file equal to the dictionary you just created
3. check that the paths in the dictionaries are correct for your workstation

also see README.md in the imabeh/run folder for run instructions
"""
import os

LOCAL_DIR, _ = os.path.split(os.path.realpath(__file__))

# location of the labserver files and data paths
GLOBAL_PATHS = {
    "labserver_files": "/mnt/upramdya_files",
    "labserver_data": "/mnt/upramdya_data",

    # paths to different folders within each trial (to append to each trial path)
    "2p_path": "/2p",
    "septacam_path": "/behData/images",
    "fictrac_path": "/behData/fictrac",
    "df3d_path": "/behData/df3d",
    "processed_path": "/processed",
    "figures_path": "/figures",

    # location of file with fly_dirs that should be processed
    "txt_file_to_process": os.path.join(LOCAL_DIR, "_fly_dirs_to_process.txt"),
    # location of the current user file
    "txt_current_user": os.path.join(LOCAL_DIR, "_current_user.txt")
}

# settins for cameras of each scope
SCOPE_CONFIG = {
    "2p_scope_1": {
        "camera_order": [0, 1, 2, 3, 4, 5, 6], # order of cameras for df3d
        "fictrac_cam": 4, # camera used for fictrac
        "opto_cam": 2     # camera with clearest laser stimulation spot
    },
    "2p_scope_2": {
        "camera_order": [6, 5, 4, 3, 2, 1, 0],
        "fictrac_cam": 3,
        "opto_cam": 5
    }
}


# default settings for each user (create a new one per person and scope)
USER_MA_2p2 = {
    "initials": "MA",
    "labserver_files": os.path.join(GLOBAL_PATHS["labserver_files"], "AZCORRA_Maite", "Experimental_data", "2P"),
    "labserver_data": os.path.join(GLOBAL_PATHS['labserver_data'], "MA"),

    # which scope you're using
    "scope": "2p_scope_2",
}


