import os
import numpy as np
import cv2
import subprocess

from imabeh.general import main


def make_video_grid(trial_dir, camera_num):
    """Create a 3x3 grid of stimulations within video
    Blank spaces if less than 9 stimulations, ignore any additional stimulations than 9.
    Add a red dot during the stimulation period.
    Pad the video with 2 seconds before and after the stimulation.
    It also compresses the video."""
    
    pad = 2  # seconds before and after the stimulations
  
    # Video path and properties
    video_path = os.path.join(trial_dir, "behData", "images", f"camera_{camera_num}.mp4")
    cap = cv2.VideoCapture(video_path)
    hz = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Get stim times
    df = main.read_main_df(trial_dir)
    opto = df['opto_stim'].values
    onsets_opto = np.where(np.diff(opto) > 0)[0][:9]
    offsets_opto = np.where(np.diff(opto) < 0)[0][:9]

    # Make sure all stim have the same duration
    median_opto_dur = int(np.median(offsets_opto - onsets_opto))
    offsets_opto = onsets_opto + median_opto_dur
    # add padding
    onsets = (onsets_opto - pad * hz).astype(int)
    dur = int(median_opto_dur + 2 * pad * hz)

    # Output video setup
    output_video_path = os.path.join(trial_dir, "processed", f"camera_{camera_num}_grid_X1.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, hz, (width * 3, height * 3))
    
    # Process snippets
    sections = []
    for i in range(9):
        if i < len(onsets):
            start = onsets[i]
            cap.set(cv2.CAP_PROP_POS_FRAMES, start)
            frames = []
            for _ in range(dur):
                ret, frame = cap.read()
                frames.append(frame)
            sections.append(frames)
        else:
            # add blank frames
            sections.append([np.zeros((height, width, 3), dtype=np.uint8) for _ in range(dur)])

    # sections is a list of len 9 (or number or stim), each element is a list of frames by time,
    # each frame containing a single image (width x height)

    # Create grid frame by frame
    for t in range(dur):
        grid_frames = []
        for j in range(3):
            row_frames = [sections[j * 3 + k][t] for k in range(3)]
            grid_frames.append(np.hstack(row_frames))
        grid_frame = np.vstack(grid_frames)

        # Add red circle during stimulation
        if pad * hz <= t < median_opto_dur + pad * hz:
            cv2.circle(grid_frame, (100, 100), 50, (0, 0, 255), -1)
        
        # save frame
        out.write(grid_frame)

    # Release resources
    cap.release()
    out.release()

    # compress video
    compressed_video_path = os.path.join(trial_dir, "processed", f"camera_{camera_num}_grid_compressed.mp4")
    # delete uncompressed video if it exists
    if os.path.exists(compressed_video_path):
        os.remove(compressed_video_path)
    _compress_video(output_video_path, compressed_video_path)

    # delete uncompressed video
    os.remove(output_video_path)




def _compress_video(input_path, output_path):
    command = [
        "ffmpeg", "-i", input_path, "-vcodec", "libx264", "-crf", "23", "-preset", "fast", output_path
    ]
    subprocess.run(command, check=True)



        