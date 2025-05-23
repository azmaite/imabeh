import os
import numpy as np
import cv2
import pickle
import shutil

from imabeh.general import main
from imabeh.run.userpaths import user_config


def split_videos(trial_dir, pad = 2.5):
    """ Split septacam videos into smaller videos based on optogenetic stimulation times.
    Each sub-video will include one stimulation trial plus "pad" seconds of video before and after 
    the stimulation.
    This script will also split the main dataframe (which includes the stimulation times etc.) into
    matching sub-dataframes.

    Parameters
    ----------
    trial_dir : str
        Path to the trial directory containing the videos and main dataframe.
    pad : float
        Number of seconds to pad before and after the stimulation times. 
        Default is 2.5 seconds (stimulation usually lasts 5 seconds, so this will give as many 
        off frames as on frames).
    """

    # Get folder containing videos
    video_dir = os.path.join(trial_dir, "behData", "images")

    # get one video to get hz
    video_path = os.path.join(video_dir, f"camera_5.mp4")
    cap = cv2.VideoCapture(video_path)
    hz = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    # Get main dataframe and stimulation times
    df = main.read_main_df(trial_dir)
    opto = df['opto_stim'].values
    onsets_opto = np.where(np.diff(opto) > 0)[0]
    offsets_opto = np.where(np.diff(opto) < 0)[0]

    # add padding
    starts = (onsets_opto - pad * hz).astype(int)
    finishes = (offsets_opto + pad * hz).astype(int)


    # Create subfolders for each of the stimulation periods (and images dir within them)
    for i in range(len(starts)):
        dir_name = video_path = os.path.join(video_dir, f"stim_{i+1}")
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)

        output_video_dir= os.path.join(video_dir, f"stim_{i+1}", 'behData')
        if not os.path.exists(output_video_dir):
            os.makedirs(output_video_dir, exist_ok=True)
        output_video_dir = os.path.join(output_video_dir, 'images')
        if not os.path.exists(output_video_dir):
            os.makedirs(output_video_dir, exist_ok=True)


    ## SPLIT VIDEOS
    # iterate across all videos
    for camera_num in range(7):

        # get video path and properties
        video_path = os.path.join(video_dir, f"camera_{camera_num}.mp4")
        cap = cv2.VideoCapture(video_path)
        hz = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # for each stimulation
        for i, (start, finish) in enumerate(zip(starts, finishes)):

            # check if the video already exists, and skip if it does
            # save within behData/images so that it can be found by fictrac and df3d
            output_video_path = os.path.join(video_dir, f"stim_{i+1}", 'behData', 'images', f"camera_{camera_num}.mp4")
            if os.path.exists(output_video_path):
                continue

            # Output video setup
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_video_path, fourcc, hz, (width, height))

            # iterate through frames between start and finish
            cap.set(cv2.CAP_PROP_POS_FRAMES, start)
            for _ in range(finish - start):
                ret, frame = cap.read()

                # save frame to new video
                out.write(frame)

            # Release resource of output video
            out.release()

        # Release resources for original video
        cap.release()
  

    ## COPY VIDEO METADATA (useful for fictrac)
    metadata_file = os.path.join(video_dir, "capture_metadata.json")
    for i in range(len(starts)):
        stim_images_dir = os.path.join(video_dir, f"stim_{i+1}", 'behData', 'images')
        shutil.copy(metadata_file, stim_images_dir)


    ## SPLIT MAIN DATAFRAME
    for i, (start, finish) in enumerate(zip(starts, finishes)):
        new_df = df[start:finish]

        # save dataframe to new video folder (for df3d and fictrac)
        stim_df_path = os.path.join(video_dir, f"stim_{i+1}", user_config["processed_path"])
        if not os.path.exists(stim_df_path):
            os.makedirs(stim_df_path)
        new_df.to_pickle(os.path.join(stim_df_path, "processed_df.pkl"))


def join_split_df(trial_dir, log):
    """ Join split dataframes into one dataframe together with the original main_df
    (which includes the info for the frames ignored during the splitting, i.e. the non-stimulation frames).
    This will be saved to the main processed folder.
    
    Parameters
    ----------
    trial_dir : str
        Path to the trial directory containing all the dataframes
   
    """

    # Get the list of split dirs in trial_dir
    video_dir = os.path.join(trial_dir, "behData", "images")
    stim_dirs = os.listdir(video_dir)
    stim_dirs = [d for d in stim_dirs if d.startswith("stim_")]

    # join each split df with the main df
    for stim_dir in stim_dirs:
        split_df_path = os.path.join(video_dir, stim_dir, user_config["processed_path"], "processed_df.pkl")

        main.combine_df(trial_dir, split_df_path, log)
        


    



