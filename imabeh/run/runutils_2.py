"""
utility functions to run processing using the TaskManager

Contains the following functions:
    

"""

import os
import smtplib
import ssl
from copy import deepcopy
from typing import List
from pexpect import pxssh

from twoppp import load
from twoppp.run.runparams import CURRENT_USER

from imabeh.run.userpaths import get_current_user_config

# get current user settings and paths
user_config = get_current_user_config()





def read_running_tasks(txt_file: str = str=user_config["_tasks_running.txt"]) -> List[dict]:
    """
    reads the supplied text file and returns a list of dictionaries
    with information for each task that is running. ONLY ONE TASK AND TRIAL EACH!
    General requested format of a line in the txt file:
    fly_dir||trial1||task1
    example:
    /mnt/nas2/JB/date_genotype/Fly1||001_beh||fictrac

    Parameters
    ----------
    txt_file : str, optional
        location of the text file, by default defined in imabeh/run/userpaths.py

    Returns
    -------
    running_tasks: List[dict]
         list of trial dict with the following fields for each fly trial:
        - "dir": the base directory of the fly
        - "trial": which trial to run on
        - "tasks": str of the task running
    """

    # read file
    with open(txt_file) as file:
        lines = file.readlines()
        lines = [line.rstrip() for line in lines]
    
    # get running tasks
    running_tasks = []
    for line in lines:
        if line.startswith("#") or line == "":
            continue
        strings = line.split("||")
        fly = {
            "dir": strings[0],
            "trial": strings[1],
            "task": strings[2],
        }
        running_tasks.append(fly)

    return running_tasks


def write_running_tasks(task: dict, add: bool = True,
                        txt_file: str = os.path.join(LOCAL_DIR, "_tasks_running.txt")) -> None:
    """
    Write or delete from the supplied text file to indicate which tasks are currently running.

    Parameters
    ----------
    task : dict
        dict with the following fields:
        - "dir": the base directory of the fly
        - "selected_trials": a string describing which trials to run on,
                             e.g. "001,002" or "all_trials"
        - "tasks": a comma separated string containing the names of the task running
    add : bool, optional
        if True: add to file. if False: remove from file, by default True
    txt_file : str, optional
        location of the text file, by default os.path.join(LOCAL_DIR, "_tasks_running.txt")
    """
    if add:
        with open(txt_file, "a") as file:
            file.write(f"\n{task['dir']}||{task['selected_trials']}||{task['tasks']}")
    else:
        with open(txt_file) as file:
            lines = file.readlines()
            lines = [line.rstrip() for line in lines]
        lines_to_write = []
        for line in lines:
            if task["dir"] in line and task["tasks"] in line:
                #TODO: check for correct selected trials
                continue
            if line == "":
                continue
            else:
                lines_to_write.append(line)
        with open(txt_file, "w") as file:
            for line in lines_to_write:
                file.write(line+"\n")


                ###############################################

def check_task_running(fly_dict: dict, task_name: str, running_tasks: List[dict]) -> bool:
    """
    search for a particular task on a particular fly in the running_tasks list to check
    whether it is running already.

    Parameters
    ----------
    fly_dict : dict
        dict with the following fields:
        - "dir": the base directory of the fly
    task_name : str
        name of the task
    running_tasks : List[dict]
        list of dictionaries specifying running tasks, each with the following entries
        - "dir": the base directory of the fly
        - "tasks": the name of the task running

    Returns
    -------
    bool
        [description]
    """
    fly_tasks = [this_task for this_task in running_tasks if this_task["dir"] == fly_dict["dir"]]
    correct_tasks = [this_task for this_task in fly_tasks if this_task["tasks"] == task_name]
    # TODO: check whether the task is running on the correct trials
    return bool(len(correct_tasks))


def send_email(subject: str, message: str, receiver_email: str) -> None:
    """
    send an email using the nelydebugging@outlook.com account

    Parameters
    ----------
    subject : str
        subject of the e-mail to be sent
    message : str
        messsage of the e-mail to be sent
    receiver_email : str
        receiver e-mail address.
    """
    smtp_server = "smtp-mail.outlook.com"
    port = 587
    sender_email = "nelydebugging@outlook.com"

    email = "Subject: " + subject + "\n\n" + message

    try:
        with open(os.path.join(LOCAL_DIR, ".pwd")) as file:
            lines = file.readlines()
            lines = [line.rstrip() for line in lines]
        password = lines[0]
    except FileNotFoundError:
        password = input("Input e-mail password. " +\
                         "To avoid entering e-mail password in the future, " +\
                         "safe it in a file called .pwd in the same folder as runutils.py.")

    context = ssl.create_default_context()

    # Try to log in to server and send email
    try:
        server = smtplib.SMTP(smtp_server, port)
        server.starttls(context=context) # Secure the connection
        server.login(sender_email, password)
        server.sendmail(sender_email, receiver_email, email)
    finally:
        server.quit()

def find_trials_2plinux(fly_dir: str, user_folder: str, twop: bool=False) -> List[str]:
    """
    find trials for a particular fly by ssh-ing into the two-photon linux computer.
    This might be useful in case the data is not copied yet completely and some trials
    only exist on the experiment computer, but not on the local computer

    Parameters
    ----------
    fly_dir : str
        base directory of the fly
    user_folder : str
        name of your user folder on the twop linux machine. usually your initials
    twop : bool, optional
        if True search for 2p data trials. If False, search for behavioural data, by default False

    Returns
    -------
    List[str]
        returns a list of trial names, i.e., the names of the folder of each trial.
    """
    ip_address = CURRENT_USER["2p_linux_ip"]
    user = CURRENT_USER["2p_linux_user"]
    try:
        with open(os.path.join(LOCAL_DIR, ".pwd")) as file:
            lines = file.readlines()
            lines = [line.rstrip() for line in lines]
        password = lines[0]
    except FileNotFoundError:
        password = input("Input 2plinux password. " +\
                         "To avoid entering 2plinux password in the future, " +\
                         "safe it in a file called .pwd in the same folder as runutils.py.")

    twop_base_dir = os.path.join("/mnt/windows_share", user_folder)
    beh_base_dir = os.path.join("/data", user_folder)
    base_dir = twop_base_dir if twop else beh_base_dir
    fly_dir_split = fly_dir.split(os.sep)
    remote_fly_dir = os.path.join(base_dir, *fly_dir_split[-2:])

    try:
        server = pxssh.pxssh()
        server.login(ip_address, user, password)
        server.sendline(f"ls {remote_fly_dir}/*/")
        server.prompt()
        answer = str(server.before)
        """
        example answer is similar to the following: (without the line breaks)
        'ls /mnt/windows_share/JB/220721_DfdxGCaMP6s_tdTom/Fly2/*/\r\n
        /mnt/windows_share/JB/220721_DfdxGCaMP6s_tdTom/Fly2/001_xz/:\r\n\x1b[0m\x1b[01;34m2p\x1b
        /mnt/windows_share/JB/220721_DfdxGCaMP6s_tdTom/Fly2/002_xz/:\r\n\x1b[01;34m2p\x1b[0m\r\n
        /mnt/windows_share/JB/220721_DfdxGCaMP6s_tdTom/Fly2/003_cc_vol/:\r\n\x1b[01;34m2p\x1b[0m
        /mnt/windows_share/JB/220721_DfdxGCaMP6s_tdTom/Fly2/004_cc_t1_vol/:\r\n\x1b[01;34m2p\x1b'
        """
        if "No such file or directory" in answer:
            server.logout()
            return []
        trial_names = answer.split(remote_fly_dir)[2:]
        trial_names = [trial_name.split(os.sep)[1] for trial_name in trial_names]
        server.logout()
        return trial_names
    except pxssh.ExceptionPxssh as error:
        print("pxssh failed while logging into the 2plinux computer.")
        print(error)
        return []