import os
import numpy as np
import cv2

from imabeh.general import main


def make_video_grid(trial_dir, camera_num):
    """ create a grid of videos for each stimulation within a trial.
    If there are 10 stimulations, it will only use the first 9 to make a 3x3 grid.
    Videos will be shown from 2s before to 2s after the stimulation."""

    pad = 2  # seconds before and after the stimulation

    # find the video path
    video_path = os.path.join(trial_dir, "behData", "images", f"camera_{camera_num}.mp4")

    # load the video and get the video properties
    cap = cv2.VideoCapture(video_path)
    hz = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # get the times for the optogenetic stimulations (only the first 9)
    df = main.read_main_df(trial_dir)
    df = main._add_attributes_to_df(df, trial_dir)
    opto = df['opto_stim'].values[0:-1]
    onsets = np.where(np.diff(opto) > 0)[0][0:9]
    offsets = np.where(np.diff(opto) < 0)[0][0:9]

    # make sure all stimulations are the same length
    median_dur = np.median(offsets - onsets).astype(int)
    offsets = onsets + median_dur

    # add the padding
    onsets = (onsets - pad*hz).astype(int)
    offsets = (offsets + pad*hz).astype(int)
    dur = np.median(offsets - onsets).astype(int)

    # Define output video properties
    output_video_path = os.path.join(trial_dir, "processed", f"camera_{camera_num}_grid.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, hz, (width * 3, height * 3))
    
    # Process each section
    sections = []
    for start_frame, end_frame in zip(onsets, offsets):
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        section_frames = []
        for _ in range(end_frame - start_frame):
            ret, frame = cap.read()
            if not ret:
                break
            section_frames.append(frame)
        sections.append(section_frames)

    # Generate the grid video
    for i in range(dur):
        grid_frames = []
        for j in range(3):
            row_frames = [cv2.resize(sections[j * 3 + k][i], (width, height)) for k in range(3)]
            grid_frames.append(np.hstack(row_frames))
        grid_frame = np.vstack(grid_frames)

        # Add a red circle in the top-left corner between stimulation onset and offset
        if pad*hz <= i < median_dur + pad*hz:
            cv2.circle(grid_frame, (90, 90), 75, (0, 0, 255), -1)

        out.write(grid_frame)
    
    # Release resources
    cap.release()
    out.release()




        