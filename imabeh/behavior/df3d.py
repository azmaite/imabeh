"""
sub-module to run pose estimation usind df3d and read the output.

Includes functions to prepare a run-script for DeepFly3D,
run said script and perform post-processing with df3dPostProcessing.
"""
import os
import pickle
import numpy as np
import pandas as pd
import sys
import cv2
import itertools
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg

# IMPORT FUNCTIONS FROM df3d (deepfly3d package) and df3dPostProcessing
from df3d.cli import main as df3dcli
from df3dPostProcessing.df3dPostProcessing import df3dPostProcess, df3d_skeleton

# IMPORT ALL PATHS FROM USERPATHS - DO NOT add any paths outside of this import 
from imabeh.run.userpaths import user_config, LOCAL_DIR

from imabeh.general.main import find_file
   

def run_df3d(trial_dir : str, video = False):
    """run deepfly3d.
    To make it faster to run, the images are copied to a local folder and df3d is run on the local folder.
    The results are then moved to the correct output directory, and the local folder is deleted.

    Parameters
    ----------
    trial_dir : str
        directory of the trial. should contain "behData/images" folder
        df3d will be saved within this trial folder as specified in the user_config
    video : bool
        whether to create a 3d video of the pose estimation
    """

    # Prepare the data for df3d by copying the images to the local data folder
    # this will make it faster to run df3d
    # output is a local folder with the images
    local_images_dir = _prepare_df3d(trial_dir)

    # Get the output_dir and camera_ids from user_config
    output_dir = user_config["df3d_path"]
    camera_ids = user_config["camera_order"]

    # Simulate the command-line arguments
    output_dir_name = 'df3d'
    if not video:
        sys.argv = [
            "df3d-cli",         # The name of the command           
            "-o", local_images_dir,  
            "--output-folder", output_dir_name,  # Temporary folder to save the results (df3d cannot save outside of images_dir)
            "--order", *map(str, camera_ids),
        ]
    else:
        sys.argv = [
            "df3d-cli",         # The name of the command           
            "-o", local_images_dir,  
            "--output-folder", output_dir_name,  # Temporary folder to save the results (df3d cannot save outside of images_dir)
            "--order", *map(str, camera_ids),
            "--video-3d",               # Generate pose3d videos
            "-n", "400"
        ]
    # Call the df3d main function to run
    # MAKE SURE YOUR .bashrc FILE HAS "export CUDA_VISIBLE_DEVICES=0" 
    # OR THE GPU WONT BE USED AND DF3D WILL BE SLOW!!!!!
    df3dcli()

    # Move the output files to the correct output directory
    output_dir = os.path.join(trial_dir, output_dir)
    try:
        os.makedirs(output_dir)
    except:
        pass
    output_dir_temp = os.path.join(local_images_dir, output_dir_name)
    for file in os.listdir(output_dir_temp):
        os.system(f"mv {os.path.join(output_dir_temp, file)} {output_dir}")
    os.rmdir(output_dir_temp)

    # delete all jpgs
    print('deleting jpgs...')
    for file in os.listdir(local_images_dir):
        if file.endswith('.jpg'):
            os.system('rm ' + os.path.join(local_images_dir, file))
    
    # Delete the local data folder created to save the videos (with all the contents)
    exp_dir = os.path.join('/', *local_images_dir.split(os.sep)[:-2])
    os.system(f"rm -r {exp_dir}")


def find_df3d_file(directory, type : str = 'result', most_recent=False):
    """
    This function finds the path to the output files of df3d.
    Can look for 'result' = (`df3d_result*.pkl`), 'angles' = (`joint_angles*.pkl`) or 'aligned' = (`aligned_pose*.pkl`)
    where `{cam}` is the values specified in the `camera` argument. 
    If multiple files with this name are found and most_recent = False, it throws an exception.
    otherwise, it returns the most recent file.

    Parameters
    ----------
    directory : str
        Directory in which to search.
    type : str
        Type of file to search for. Can be 'result', 'angles' or 'aligned'.
    most_recent : bool
        if True, returns the most recent file if multiple are found, by default False

    Returns
    -------
    path : str
        Path to df3d output file.

    """
    search_type = {'result': 'df3d_result_*.pkl', 'angles': 'joint_angles_*.pkl', 'aligned': 'aligned_pose_*.pkl'}
    # check that the type is correct
    if type not in search_type.keys():
        raise ValueError(f"Type must be one of {list(search_type.keys())}")
    
    # find file
    return find_file(directory,
                      search_type[type],
                      "df3d output " + type,
                      most_recent=most_recent)

def postprocess_df3d_trial(trial_dir):
    """run post-processing of deepfly3d data as defined in the df3dPostProcessing package:
    Align with reference fly template and calculate leg angles.

    Parameters
    ----------
    trial_dir : string
        directory of the trial. should contain an "images" folder at some level of hierarchy
    """
    # get the path to the (most recent) df3d result file
    df3d_dir = os.path.join(trial_dir, user_config["df3d_path"])
    pose_result = find_df3d_file(df3d_dir, 'result', most_recent=True)
    pose_result_name = "df3d_result"

    # run df3d post-processing
    mydf3dPostProcess = df3dPostProcess(exp_dir=pose_result, calculate_3d=True, outlier_correction=True)
    
    # align model and save
    aligned_model = mydf3dPostProcess.align_to_template(interpolate=False, scale=True, all_body=True)
    path = pose_result.replace(pose_result_name,'aligned_pose')
    with open(path, 'wb') as f:
        pickle.dump(aligned_model, f)

    # calculate leg angles and save
    leg_angles = mydf3dPostProcess.calculate_leg_angles(save_angles=False)
    path = pose_result.replace(pose_result_name, 'joint_angles')
    with open(path, 'wb') as f:
        pickle.dump(leg_angles, f)

def get_df3d_df(trial_dir):
    """load pose estimation data into a dataframe.
    Adds columns for joint position and joint angles in the final format
    (one column per leg joint angle/position)

    Parameters
    ----------
    trial_dir : str
        base directory where pose estimation results can be found

    Returns
    -------
    beh_df_path: str
        Path to df3d dataframe containing pose estimation data
    """
    # get the path to the most recent df3d result file as well as aligned and angles files
    df3d_dir = os.path.join(trial_dir, user_config["df3d_path"])
    df3d_result = find_df3d_file(df3d_dir, 'result', most_recent=True)
    df3d_angles = find_df3d_file(df3d_dir, 'angles', most_recent=True)
    df3d_aligned = find_df3d_file(df3d_dir, 'aligned', most_recent=True)

    # read the angles and joit positions and convert naming to final format
    # currently one dictionary per leg/region with subdictionary for each angle/position
    # new format: one dictionary per leg/region and angle/position

    # make empty dictionary to store data
    df3d_dict = {}

    # read angles file
    with open(df3d_angles, "rb") as f:
        angles = pickle.load(f)
    # get leg and angle keys
    leg_keys = [key for key in angles.keys()]
    angle_keys = [key for key in angles[leg_keys[0]].keys()]
    # add joint angles with proper names and number format to df3d_dict
    for leg in leg_keys:
        if leg == "Head": # skip head, empty (no angles)
            continue
        for angle in angle_keys:
            new_name = "angle_" + leg + "_" + angle
            new_vals = np.array(angles[leg][angle])
            df3d_dict[new_name] = new_vals

    # read the joint position file
    with open(df3d_aligned, "rb") as f:
        joints = pickle.load(f)
    # get leg keys
    leg_keys = [key for key in joints.keys()]
    # add positions proper names and number format to df3d_dict
    # for each joint, iterate through x,y,z too
    for leg in leg_keys:
        joint_keys = [key for key in joints[leg].keys()]
        for joint in joint_keys:
            for i_xyz, xyz in enumerate(["x", "y", "z"]):
                new_name = "joint" + leg + "_" + joint + "_" + xyz
                new_vals = np.array(joints[leg][joint]["raw_pos_aligned"][:, i_xyz])
                df3d_dict[new_name] = new_vals

    # get the abdomen positions from df3d result file
    with open(df3d_result, "rb") as f:
        pose = pickle.load(f)
    # get abdomen indeces from df3d_skeleton
    abdomen_keys = ["RStripe1", "RStripe2", "RStripe3", "LStripe1", "LStripe2", "LStripe3"]
    # add abdomen positions with proper names and number format to df3d_dict
    for i_abd, abd_key in enumerate(abdomen_keys):
        for i_xyz, xyz in enumerate(["x", "y", "z"]):
            new_name = "joint_Abd_" + abd_key + "_" + xyz
            new_vals = np.array(pose["points3d"][:, i_abd, i_xyz])
            df3d_dict[new_name] = new_vals

    # create a dataframe with all the data
    df3d_df = pd.DataFrame(df3d_dict)

    # save the dataframe in the same folder as the df3d results as a pickle file
    df_out_dir = os.path.join(os.path.dirname(df3d_result), "df3d_df.pkl")
    df3d_df.to_pickle(df_out_dir)
    
    return df_out_dir


def df3d_video(trial_dir : str, start_frame : int = 0, end_frame : int = 100):
    """ make a video of the 2d and 3d pose estimation after PostProcessing.

    Parameters
    ----------
    trial_dir : str
        base directory where pose estimation results can be found
    start_frame : int
        starting frame for the video
    end_frame : int
        ending frame for the video
    """

    # get the df3d result file (after post-processing) - including 2d and 3d results
    df3d_dir = trial_dir + '/behData/df3d'
    df3d_result_path = find_df3d_file(df3d_dir, 'result', most_recent=True)
    df3d_result = pickle.load(open(df3d_result_path, 'rb'))
    points2d = df3d_result["points2d"]

    # get the camera order from user_config and remove the front camera
    cameras = [int(i) for i in user_config["camera_order"]]
    cameras = cameras[::-1]

    ## make the 2d videos by stacking videos for all left legs on top 
    ## and right legs on the bottom and adding the 2d result points

    # get the shape and hz of the videos
    video_path = os.path.join(trial_dir, 'behData', 'images', 'camera_5.mp4')
    cap = cv2.VideoCapture(video_path)
    hz = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()


    ## MAKE 2D VIDEOS
    # Get video snippets for each camera in order
    videos = []
    for camera_idx, camera_num in enumerate(cameras):
        # skip the front camera
        if camera_num == 3:
            continue

        # set the colors
        if camera_idx < 3:
            RL = 'L'
        else:
            RL = 'R'

        # get the video file
        video_path = os.path.join(trial_dir, 'behData', 'images', f'camera_{camera_num}.mp4')
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        # add the frames from start to end_frame to videos list
        frames = []
        for frame_num in range(end_frame - start_frame):
            _, frame = cap.read()

            # plot 2d points on the frame
            frame = _plot_df3d_2d(frame, points2d, camera_idx, frame_num + start_frame, RL, width, height)

            # add frame to the list of frames for each video
            frames.append(frame)

        # add the camera frames to the videos list
        videos.append(frames)

        # release resources
        cap.release()


    ## Make the 3d videos and append
    videos_3d = _make_df3d_3d(trial_dir, start_frame, end_frame, width)
    videos.append(videos_3d[0])
    videos.append(videos_3d[1])
    videos.append(videos_3d[2])

    # crop the 3d videos (too much black on top)
    for i in range(6,9):
        for t in range(len(videos[i])):
            videos[i][t] = videos[i][t][200:,:,:]


    # Output video setup
    output_video_path = os.path.join(trial_dir, "behData", "df3d", f"df3d_video_3d.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, hz/4, (width*3, height*3+280)) #4 times slowed down

    # Create grid frame by frame
    for t in range(end_frame - start_frame - 1):
        # stack the frames for each camera row by row
        grid_frames = []
        for j in range(3): # 3 rows
            row_frames = [videos[j * 3 + k][t] for k in range(3)]
            grid_frames.append(np.hstack(row_frames))
        grid_frame = np.vstack(grid_frames)
        # save frame
        out.write(grid_frame)

    # Release resources
    out.release()

    return videos, grid_frame





    



## Helper functions

def _prepare_df3d(trial_dir : str):
    """ prepare to run df3d by copying the videos from the server to the local machine
    
    Parameters
    ----------
    trial_dir : str
        directory of the trial. should contain "behData/images" folder
        df3d will be saved within this trial folder as specified in the user_config
    
    """
    # get the path to the local data folder to copy videos to (and create if doesn't exist)
    local_data = user_config["local_data"]
    os.makedirs(local_data, exist_ok=True)

    # get the path to the images folder
    images_dir = os.path.join(trial_dir, "behData", "images")

    # create a temporary folder to save the images within local data folder
    new_folder = os.path.join(*images_dir.split(os.sep)[-3:-1])
    local_data_folder = os.path.join(local_data, new_folder)
    os.makedirs(local_data_folder, exist_ok=True)
    os.system(f"cp -r {images_dir} {local_data_folder}")
    
    return os.path.join(local_data_folder, 'images')


def _plot_df3d_2d(frame, points2d, camera_idx, frame_num, RL, width, height):
    
    # set the colors for the left and right legs, and leg/abdomen names
    legs_abd = ['F', 'M', 'H', 'Stripe'] # strip is for abdomen, 1, 2, 3
    colors = {'L':[[15, 115, 153], [26, 141, 175], [117, 190, 203]], 'R':[[186, 30, 49], [201, 86, 79], [213, 133, 121]]} # LF, LM, LH, RF, RM, RH
    leg_segments = ['Coxa', 'Femur', 'Tibia', 'Tarsus', 'Claw']
    # convert all colors to BGR (OpenCV uses BGR)
    for k in colors.keys():
        for i in range(len(colors[k])):
            colors[k][i] = colors[k][i][::-1]

    # get colors
    colors_RL = colors[RL]

    for leg_idx, leg in enumerate(legs_abd):
        # get the leg points for each leg segment or abd
        leg_points_idx = []
        if leg == 'Stripe':
            segments = ['1', '2', '3']
        else:
            segments = leg_segments

        for segment in segments:
            # get the index for each leg segment
            leg_point_name = RL + leg + segment
            leg_point_idx = df3d_skeleton.index(leg_point_name)
            leg_points_idx.append(leg_point_idx)
        # get the 2d points for each leg segment (x,y)
        leg_points = points2d[camera_idx, frame_num, leg_points_idx, :]
        # get the color to plot
        if leg == 'Stripe':
            color = [200, 200, 200]
        else:
            color = colors_RL[leg_idx]

        # Convert to absolute pixel coordinates
        xy = np.transpose(np.vstack([leg_points[:, 1] * width, leg_points[:, 0] * height]))

        # Ensure integer type and correct shape for OpenCV
        xy = xy.astype(np.int32).reshape((-1, 1, 2))

        # Draw the polyline for each leg/abd
        cv2.polylines(frame, [xy], isClosed=False, color=color, thickness=7)
        # Draw circles at each vertex
        for point in xy:
            center = tuple(point[0])  # Extract (x, y) tuple
            cv2.circle(frame, center, radius=5, color=(128, 0, 0), thickness=5)  # Navy blue empty circles

    return frame


def _make_df3d_3d(trial_dir, start_frame, end_frame, width):

    # load aligned 3d points
    df3d_dir = trial_dir + '/behData/df3d'
    df3d_aligned = find_df3d_file(df3d_dir, 'aligned', most_recent=True)
    df3d_aligned = pickle.load(open(df3d_aligned, 'rb'))

    # set the colors for the left and right legs, and leg names
    legs = ['F', 'M', 'H']
    sides = ['R','L']
    colors = {'L':[[15, 115, 153], [26, 141, 175], [117, 190, 203]], 'R':[[186, 30, 49], [201, 86, 79], [213, 133, 121]]} # LF, LM, LH, RF, RM, RH
    leg_segments = ['Coxa', 'Femur', 'Tibia', 'Tarsus', 'Claw']
    # convert all colors to BGR (OpenCV uses BGR)
    for k in colors.keys():
        for i in range(len(colors[k])):
            colors[k][i] = colors[k][i][::-1]

    videos = [[], [], []]
    for frame_num in range(start_frame, end_frame):

        # make empty 3d figure
        fig = plt.figure(figsize=(4,4))
        ax = fig.add_subplot(111, projection='3d')

        # plot legs
        for leg, side in itertools.product(legs, sides):
            legRL = side + leg
            leg_points = []
            for segment in leg_segments:
                xyz = df3d_aligned[f'{legRL}_leg'][segment]['raw_pos_aligned'][frame_num]
                leg_points.append(xyz)
            leg_points = np.vstack(leg_points)

            # plot
            color = np.divide(colors[side][legs.index(leg)], 255)
            ax.plot(leg_points[:, 0], leg_points[:, 1], leg_points[:, 2], color=color, linewidth=3)

        # plot abdomen
        #TODO

        ## STYLE OF PLOT
        # Remove grid and make background black
        ax.set_facecolor('black')  # Set figure background
        ax.grid(False)  # Remove grid

        # Hide axes and ticks
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_zlabel('')

        # Set limits
        ax.set_xlim([-2, 2])
        ax.set_ylim([-2, 2])
        ax.set_zlim([-2, 0.5])

        # Make back panes black
        ax.xaxis.set_pane_color((0, 0, 0, 1))
        ax.yaxis.set_pane_color((0, 0, 0, 1))

        # set view angle (x3) - and save each view
        angles = [70, 120, 170]
        for i, angle in enumerate(angles):
            ax.view_init(elev=30, azim=angle)

            # Save the figure as an image
            converted_img = _figure_to_image(fig, width)
            videos[i].append(converted_img)

        plt.close(fig)  # Close figure so it doesn't show in interactive environments

    # return images
    return videos


def _figure_to_image(fig, width):
    """Convert a Matplotlib figure to a NumPy RGB image without a white border."""
    fig.patch.set_alpha(0)  # Ensure the figure background is transparent
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)  # Remove padding
    
    # Draw figure
    fig.canvas.draw()
    
    # Convert to NumPy array
    img = np.array(fig.canvas.renderer.buffer_rgba())
    
    # Remove alpha channel (RGBA -> RGB)
    img = img[:, :, :3]

    # Resize to match the desired width while maintaining aspect ratio
    aspect_ratio = img.shape[1] / img.shape[0]
    new_height = int(width / aspect_ratio)
    img_resized = cv2.resize(img, (width, new_height))

    return img_resized