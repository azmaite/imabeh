"""
sub-module to run and analyse fictrac.

Includes functions to prepare the required config file and run ficrac in a new process.
Includes functionality to read results from fictrac & combine them with an existing Pandas dataframe
Copied and slightly modified from NeLy-EPFL/twoppp/behavior/fictrac.py

Contains functions:

# MAIN FUNCTION TO CONFIG AND RUN FICTRAC
- config_and_run_fictrac()

# ACCESSORY FUNCTIONS USED IN config_and_run_fictrac
- _get_mean_image()
- _get_ball_parameters()
- _get_circ_points_for_config()
- _write_config_file()
- _run_fictrac_config_gui()
- _run_fictrac()

# FUNCTIONS TO READ FICTRAC OUTPUT
- _get_septacam_fps()
- _filter_fictrac()
- get_fictrac_df()
"""

import os
import numpy as np
import pandas as pd
import cv2
import json
import glob
from scipy.ndimage import gaussian_filter1d, median_filter

from imabeh.run.userpaths import user_config, LOCAL_DIR
from imabeh.behavior import behaviormain
from imabeh.general import main


# set ball radius in mm
R_BALL = 5 

# set column names to read fictrac output
# see https://github.com/rjdmoore/fictrac/blob/master/doc/data_header.txt for fictrac output description
COL_NAMES = ["Frame_counter",
             "delta_rot_cam_right", "delta_rot_cam_down", "delta_rot_cam_forward",
             "delta_rot_error",
             "delta_rot_lab_side", "delta_rot_lab_forward", "delta_rot_lab_turn",
             "abs_rot_cam_right", "abs_rot_cam_down", "abs_rot_cam_forward",
             "abs_rot_lab_side", "abs_rot_lab_forward", "abs_rot_lab_turn",
             "integrated_lab_x", "integrated_lab_y",
             "integrated_lab_heading",
             "animal_movement_direction_lab",
             "animal_movement_speed",
             "integrated_forward_movement", "integrated_side_movement",
             "timestamp",
             "seq_counter",
             "delta_time",
             "alt_time"
            ]

# get camera number from user_config
CAMERA_NUM = user_config['fictrac_cam']

## MAIN FUNCTION TO CONFIG AND RUN FICTRAC

def config_and_run_fictrac(trial_dir):
    """Automatically create config file for fictrac and then run it using the newly generated config.
    save everything in folder given by user_config['fictrac_path'].

    Parameters
    ----------
    trial_dir : string
        absolute directory pointing to a trial directory.
    """
    # get video file name from user_config
    video_file = main.find_file(trial_dir, f"camera_{CAMERA_NUM}.mp4", file_type="video")
    # get the fictrac output directory from user_config and trial_dir
    output_dir = os.path.join(trial_dir, user_config['fictrac_path'])
    # check if dir already exists. If not, create it
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    # check that video exists
    if not os.path.isfile(video_file):
        FileNotFoundError(f"Could not find video file: {video_file}.")

    # get mean image, ball parameters, and create config file
    mean_image = _get_mean_image(video_file, output_dir=output_dir)
    x_min, y_min, r_min = _get_ball_parameters(mean_image, output_dir=output_dir)
    points = _get_circ_points_for_config(x_min, y_min, r_min, img_shape=mean_image.shape[:2])
    config_file = _write_config_file(video_file, output_dir, points, overwrite=False)
    # run config gui
    success = _run_fictrac_config_gui(config_file)
    if not success:
        RuntimeError("Fictrac config gui failed.")

    # run fictrac and save output automatically
    success = _run_fictrac(config_file)
    if not success:
        print('why') #FOR SOME REASON THIS WORKS BUT NOT THE ERROR BELOW!!!
        RuntimeError("Fictrac running failed.")

    # move the output file to the output directory and delete fictrac log files
    _move_fictrac_output(video_file, output_dir)
    


## ACCESSORY FUNCTIONS USED IN config_and_run_fictrac

def _get_mean_image(video_file, output_dir, skip_existing=True):
    """compute the mean image of a video and save it as a file.

    Parameters
    ----------
    trial_dir : string
        absolute path of the trial directory
    video_file : string
        relative path of the video file within the trial directory
    skip_existing : bool, optional
        if already computed, read the image and return it, by default True

    Returns
    -------
    numpy array
        mean image (greyscale)
    """

    # generate file saving path
    output_name = "_".join(os.path.basename(os.path.normpath(video_file)).split("_")[:-1]) + "_mean_image.jpg"
    mean_frame_file = os.path.join(output_dir, output_name)

    # if already computed, return the image
    if skip_existing and os.path.isfile(mean_frame_file):
        print(f"{mean_frame_file} exists loading image from file without recomputing.")
        mean_frame = cv2.imread(mean_frame_file)[:, :, 0]

    else:
        # load video frame by frame
        f = cv2.VideoCapture(video_file)
        rval, frame = f.read()
        # Convert rgb to grey scale and sum all frames
        frame_sum = np.zeros_like(frame[:, :, 0], dtype=np.int64)
        count = 0
        while rval:
            frame_sum =  frame_sum + frame[:, :, 0]
            rval, frame = f.read()
            count += 1
        f.release()
        # get mean of all frames
        mean_frame = frame_sum / count
        mean_frame = mean_frame.astype(np.uint8)
        # save mean image
        cv2.imwrite(mean_frame_file, mean_frame)

    return mean_frame

def _get_ball_parameters(img, output_dir=None):
    """Using an image that includes the ball, for example the mean image,
    compute the location and the radius of the ball.
    Uses cv2.HoughCircles to find circles in the image and then selects the most likely one.

    Parameters
    ----------
    img : np.array
        image to be analysed

    output_dir : string, optional
        if specified, make image that includes the analysis results and save to file,
        by default None

    Returns
    -------
    float
        x position in pixels
    float
        y position in pixels
    float
        radius in pixels
    """
    # blur image
    img = cv2.medianBlur(img, 5)
    # set parameters
    canny_params = dict(threshold1 = 120, threshold2 = 60)  # Florian's original params: 40 & 50
    # find edges
    edges = cv2.Canny(img, **canny_params)
    # find circles
    circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 2, 200, param1=120, param2=10, minRadius=200, maxRadius=300)
    
    # select most likely circle
    if circles is not None:
        circles = np.round(circles[0, :]).astype(int)
        inside = np.inf
        x_min, y_min, r_min = np.nan, np.nan, np.nan
        for x, y, r in circles:
            if x + r > img.shape[1] or x - r < 0:  # check that ball completely in the image in x
                continue
            elif x < img.shape[1] * 1 / 4 or x > img.shape[1] * 3 / 4:  # check that ball center in central half of x axis
                continue
            elif y - r <= img.shape[0] / 10:  # check that top of the ball is below 1/10 of the image
                continue
            xx, yy = np.meshgrid(np.arange(img.shape[1]), np.arange(img.shape[0]))
            xx = xx - x
            yy = yy - y
            rr = np.sqrt(xx ** 2 + yy ** 2)
            mask = (rr < r)
            current_inside = np.mean(edges[mask])  # np.diff(np.quantile(edges[mask], [0.05, 0.95]))
            if  current_inside < inside:
                x_min, y_min, r_min = int(x), int(y), int(r)
                inside = current_inside

        # save image with circle if requested
        if output_dir is not None:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            for x, y, r in circles:
                cv2.circle(img, (x, y), r, (255, 255, 255), 1)
                cv2.rectangle(img, (x - 5, y - 5), (x + 5, y + 5), (255, 128, 255), -1)
            cv2.circle(img, (x_min, y_min), r_min, (255, 0, 0), 1)
            cv2.rectangle(img, (x_min - 5, y_min - 5), (x_min + 5, y_min + 5), (255, 128, 255), -1)
            cv2.imwrite(os.path.join(output_dir, "camera_3_circ_fit.jpg"), img)

        # return ball location and radius
        return x_min, y_min, r_min

def _get_circ_points_for_config(x, y, r, img_shape, n=12):
    """convert circle parameters into individual points on the surface of the ball
    as if they were generated from the fictrac config gui

    Parameters
    ----------
    x : float
        x position of ball in pixels
    y : float
        y position of ball in pixels
    r : float
        radius of ball in pixels
    img_shape : tuple/list
        shape of the image as (y, x)
    n : int, optional
        number of points, by default 12

    Returns
    -------
    list
        points on the ball surface, to be handed over to write_config_file()
    """
    # Compute angular limit given by image size
    theta1 = np.arcsin((img_shape[0] - y) / r)
    theta2 = 1.5 * np.pi - (theta1 - 1.5 * np.pi)

    points = []
    for theta in np.linspace(theta1, theta2, n):
        point_x = x - np.cos(theta) * r
        point_y = y - np.sin(theta) * r
        points.append(int(point_x))
        points.append(int(point_y))

    return points

def _format_list(l):
    """format a list as a string in a format that is suitable for the fictrac config file

    Parameters
    ----------
    l : list

    Returns
    -------
    string
    """
    s = repr(l)
    s = s.replace("[", "{ ")
    s = s.replace("]", " }")
    return s

def _write_config_file(video_file, output_dir, roi_circ, vfov=3.05, q_factor=40, c2a_src="c2a_cnrs_xz", do_display="n",
                      c2a_t=[-5.800291, -23.501165, 1762.927645], c2a_r=[1.200951, -1.196946, -1.213069],
                      c2a_cnrs_xz=[422, 0, 422, 0, 422, 10, 422, 10], overwrite=False, ignore_roi=None):
    """Create a config file for fictrac.
    See: https://github.com/rjdmoore/fictrac/blob/master/doc/params.md for interpretation of parameters

    Parameters
    ----------
    video_file : string
        absolute path of video file to run fictrac on
    roi_circ : list
        points on the circumference of the ball defining the ball.
        can be generated using get_circ_points_for_config()
    vfov : float, optional
        [description], by default 3.05
    q_factor : int, optional
        quality factor of fictrac, by default 40
    c2a_src : str, optional
        [description], by default "c2a_cnrs_xz"
    do_display : str, optional
        [description], by default "n"
    c2a_t : list, optional
        [description], by default [-5.800291, -23.501165, 1762.927645]
    c2a_r : list, optional
        [description], by default [1.200951, -1.196946, -1.213069]
    c2a_cnrs_xz : list, optional
        [description], by default [422, 0, 422, 0, 422, 10, 422, 10]
    overwrite : bool, optional
        whether to overwrite an existing config file, by default False
    ignore_roi : list, optional
        list of points defining the ROI to be ignored by Fictrac, by default see below
        refers to the part of the ball obscructed by the fly

    Returns
    -------
    string
        location of config file
    """

    # get default ignore_roi
    if ignore_roi is None:
        ignore_roi = [579, 122, 528, 141, 477, 134, 438, 140, 381, 151, 323, 153, 291, 134, 234, 75, 320, 39, 323, 37, 416, 27, 499, 27, 568, 26]

    # check if config file already exists
    config_file = os.path.join(output_dir, "config.txt")
    if not overwrite and os.path.isfile(config_file):
        print(f"Not writing to {config_file} because it exists.")
        return config_file

    # write config file
    content = f"vfov             : {vfov:.2f}"
    content += f"\nsrc_fn           : {video_file}"
    content += f"\nq_factor         : {int(q_factor)}"
    content += f"\nc2a_src          : {c2a_src}"
    content += f"\ndo_display       : {do_display}"
    content += f"\nroi_ignr         : {{ {_format_list(ignore_roi)} }}"
    content += f"\nc2a_t            : {_format_list(c2a_t)}"
    content += f"\nc2a_r            : {_format_list(c2a_r)}"
    content += f"\nc2a_cnrs_xz      : {_format_list(c2a_cnrs_xz)}"
    content += f"\nroi_circ         : {_format_list(roi_circ)}"
    
    with open(config_file, "w", encoding="utf-8") as f:
        f.write(content)

    return config_file

def _run_fictrac_config_gui(config_file, fictrac_config_gui="~/bin/fictrac/bin/configGui"):
    """runs the fictrac config gui in a subprocess and sequentially sends "y\n" responses to continue.
    This is required because the config gui computes some parameters based on the inputs given.

    Parameters
    ----------
    config_file : str
        absolut path of config file (already in the fictrac output directory)
    fictrac_config_gui : str, optional
        location of fictrac config gui command, by default "~/bin/fictrac/bin/configGui"
         
    Returns
    -------
    success : bool
        whether fictrac config gui was run successfully
    """
    directory = os.path.dirname(config_file)
    command = f'/bin/bash -c "cd {directory} && yes | xvfb-run -a {fictrac_config_gui} {config_file}"'
    success = main.run_shell_command(command, allow_ctrl_c=False, suppress_output=True)

    return success

def _run_fictrac(config_file, fictrac="~/bin/fictrac/bin/fictrac"):
    """Runs fictrac in the current console using the subprocess module.
    The console will not be blocked, but the outputs will be printed regularily

    Parameters
    ----------
    config_file : str
        path to config file generate by the config gui or automatically
    fictrac : str, optional
        location of fictrac on computer, by default "~/bin/fictrac/bin/fictrac"
    
    Returns
    -------
    success : bool
        whether fictrac was run successfully
    """
    command = f"{fictrac} {config_file}"
    success = main.run_shell_command(command, allow_ctrl_c=True, suppress_output=False)
    print(f"success = {success}")
    return success

def _move_fictrac_output(video_file, output_dir):
    """move the output of fictrac to the correct directory.
    Automatically it gets saved in the video directory instead of the output dir.
    Also removes the log files that fictrac creates.

    Parameters
    ----------
    video_file : string
        path to video file where fictract output is automatically saved
    output_dir : string
        path to output directory of choice
    """
    # get the dir where the video is (and where the output is saved)
    video_dir = os.path.dirname(video_file)

    # find the output files' full names
    camera = f"camera_{CAMERA_NUM}"
    output_files = []
    output_files.append(glob.glob(os.path.join(video_dir, camera + "*.dat"))[0])
    output_files.append(glob.glob(os.path.join(video_dir, camera + "-configImg.png"))[0])
    output_files.append(glob.glob(os.path.join(video_dir, camera + "-template.png"))[0])
    
    # move the output files to the output directory
    for file in output_files:
        new_name = os.path.join(output_dir, os.path.basename(file))
        os.rename(file, new_name)
    
    # remove fictrac log files in imabeh/run
    log_files = glob.glob(os.path.join(LOCAL_DIR, "fictrac*.log"))
    for file in log_files:
        os.remove(file)


## FUNCTIONS TO READ FICTRAC OUTPUT
    
def _get_septacam_fps(trial_dir):
    """get the fps of the septacam from the metadata file

    Parameters
    ----------
    trial_dir : str
        trial directory
    
    Returns
    -------
    f_s : int
        frame rate of the septacam as given by the metadata file
    """   

    cam_metadata_file = behaviormain.find_seven_camera_metadata_file(trial_dir)
    metadata = json.load(open(cam_metadata_file))
    f_s = metadata['FPS']

    return f_s

def _filter_fictrac(time_series, med_filt_size=5, sigma_gauss_size=10):
    """apply Median filter and Gaussian filter to fictrac time series

    Parameters
    ----------
    time_series : numpy array
        time series to filter
    med_filt_size : int, optional
        size of median filter, by default 5
    sigma_gauss_size : int, optional
        width of Gaussian kernel, by default 10

    Returns
    -------
    numpy array
        filtered time series
    """
    return gaussian_filter1d(median_filter(time_series, size=med_filt_size), sigma=sigma_gauss_size)

def get_fictrac_df(trial_dir, med_filt_size=5, sigma_gauss_size=10):
    """Read the output of fictrac, convert it into physical units and save it in a dataframe.

    Parameters
    ----------
    trial_dir : str
        trial directory
    med_filt_size : int, optional
        size of median filter applied to velocity and orientation, by default 5
    sigma_gauss_size : int, optional
        width of Gaussian kernel applied to velocity and orientation, by default 10

    Returns
    -------
    fictact_df_path : str
        path to the dataframe
    """
    # get septacam frame rate from meatatada
    f_s = _get_septacam_fps(trial_dir)

    # find the most recent fictrac output file
    fictrac_output = behaviormain.find_fictrac_file(trial_dir, camera=CAMERA_NUM, most_recent=True)
    # read the data
    fictrac_df = pd.read_csv(fictrac_output, header=None, names=COL_NAMES)
    
    # convert time series to physical units (using framerate and ball radius)
    fictrac_df["v_raw"] = fictrac_df["animal_movement_speed"] * f_s * R_BALL # convert from rad/frame to rad/s and mm/s
    fictrac_df["th_raw"] = (fictrac_df["animal_movement_direction_lab"] - np.pi) / np.pi * 180
    fictrac_df["x"] = fictrac_df["integrated_lab_x"] * R_BALL
    fictrac_df["y"] = fictrac_df["integrated_lab_y"] * R_BALL
    fictrac_df["integrated_forward_movement"] *=  R_BALL
    fictrac_df["integrated_side_movement"] *=  R_BALL
    fictrac_df["delta_rot_lab_side"] *= R_BALL * f_s
    fictrac_df["delta_rot_lab_forward"] *= R_BALL * f_s
    fictrac_df["delta_rot_lab_turn"] *= R_BALL * f_s / np.pi * 180
    
    # filter velocity and orientation time series
    fictrac_df["v"] = _filter_fictrac(fictrac_df["v_raw"], med_filt_size, sigma_gauss_size)
    fictrac_df["th"] = _filter_fictrac(fictrac_df["th_raw"], med_filt_size, sigma_gauss_size)
    fictrac_df["v_forw"] = _filter_fictrac(fictrac_df["delta_rot_lab_forward"], med_filt_size, sigma_gauss_size)
    fictrac_df["v_side"] = _filter_fictrac(fictrac_df["delta_rot_lab_side"], med_filt_size, sigma_gauss_size)
    fictrac_df["v_turn"] = _filter_fictrac(fictrac_df["delta_rot_lab_turn"], med_filt_size, sigma_gauss_size)
    
    # reorganize columns
    fictrac_df = fictrac_df[["v_raw", "th_raw", "x", "y", "integrated_forward_movement",
                             "integrated_side_movement", "delta_rot_lab_side",
                             "delta_rot_lab_forward", "delta_rot_lab_turn", "v", "th",
                             "v_forw", "v_side", "v_turn"]]
    
    # save the dataframe in the same folder as the fictrac output as a pickle file
    df_out_dir = os.path.join(os.path.dirname(fictrac_output), "fictrac_df.pkl")
    fictrac_df.to_pickle(df_out_dir)
    
    return df_out_dir
