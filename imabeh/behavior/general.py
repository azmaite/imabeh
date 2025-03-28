import os
import numpy as np
import cv2

from imabeh.general import main
from imabeh.run.userpaths import user_config


def split_videos(trial_dir, pad = 2.5):
    """ Split septacam videos into smaller videos based on optogenetic stimulation times.
    Each sub-video will include one stimulation trial plus "pad" seconds of video before and after 
    the stimulation.
    This script will also split the main dataframe (which includes the stimulation times etc.) into
    matching sub-dataframes.
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
    onsets_opto = np.where(np.diff(opto) > 0)[0][:9]
    offsets_opto = np.where(np.diff(opto) < 0)[0][:9]

    # add padding
    starts = (onsets_opto - pad * hz).astype(int)
    finishes = (offsets_opto + pad * hz).astype(int)

    # Create subfolders for each of the stimulation periods
    for i in range(len(starts)):
        dir_name = video_path = os.path.join(video_dir, f"stim_{i+1}")
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)

    # iterate across all videos
    for camera_num in range(7):

        # get video path and properties
        video_path = os.path.join(video_dir, f"camera_{camera_num}.mp4")
        cap = cv2.VideoCapture(video_path)
        hz = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # for each stimulation
        for start, finish in zip(starts, finishes):

            # Output video setup
            output_video_path = os.path.join(video_dir, f"stim_{i+1}", f"camera_{camera_num}_stim_{i+1}.mp4")
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


    # Split df dataframe too and save
    df_dir = os.path.join(trial_dir, user_config["processed_path"])#, "processed_df.pkl")
    for i, (start, finish) in enumerate(zip(starts, finishes)):
        new_df = df[start:finish]
        new_df.to_pickle(os.path.join(df_dir, f"processed_df_stim_{i+1}.pkl"))

    






    # # run df3d, postprocess and get df
    # df3d.run_df3d(trial_dir)
    # df3d.postprocess_df3d_trial(trial_dir)
    # _ = df3d.get_df3d_df(trial_dir)

    # # run fictrac and convert output to df
    # fictrac.config_and_run_fictrac(trial_dir)
    # _ = fictrac.get_fictrac_df(trial_dir)
        
