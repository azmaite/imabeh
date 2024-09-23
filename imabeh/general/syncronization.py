"""
Synchronization module
======================

This module provides functions to process the synchronization data
acquired with Thor Sync during imaging and behavior

COPIED FROM NeLy-EPFL/utils2p
Only relevant functions are kept. 
Changed examples etc.
Also some changes to the original functions:
- removed .astype(int) in returns (was giving errors and was already int or nan)
- changed get_processed_lines: original did not work if imaging and/or behavior were not present
- added _get_appropriate_lines: supporting function for get_processed_lines, faster than before as it knows the thorsync structure
"""

import warnings
import json

import numpy as np
import h5py

from imabeh.imaging2p import utils2p
from imabeh.run.userpaths import user_config


class SynchronizationError(Exception):
    """The input data is not consistent with synchronization assumption."""


def get_times(length, freq):
    """
    This function returns the time point of each tick
    for a given sequence length and tick frequency.

    Parameters
    ----------
    length : int
        Length of sequence.
    freq : float
        Frequency in Hz.

    Returns
    -------
    times : array
        Times in seconds.
    """
    times = np.arange(0, length / freq, 1 / freq)
    return times


def edges(line, size=0, correct_possible_split_edges=True):
    """
    Returns the indices of edges in a line. An edge is change in value of
    the line. A size argument can be specified to filter for changes of
    specific magnitude. By default only rising edges (increases in value)
    are returned.

    Parameters
    ----------
    line : numpy array
        Line signal from h5 file.
    size : float or tuple
        Size of the rising edge. If float it is used as minimum.
        Tuples specify a range. To get falling edges use negative values.
        Only one boundary can be applied using np.inf as one of the values.
        All boundaries are excluding the specified value.
    correct_possible_split_edges : boolean
        The rise or fall of an edge can in some cases be spread over
        several ticks. If `True` these "blurry" edges are sharpened
        with :func:`utils2p.synchronization.correct_split_edges`.
        Default is True.

    Returns
    -------
    indices : list
        Indices of the rising edges.

    Examples
    --------
    >>> binary_line = np.array([0, 1, 1, 0, 1, 1])
    >>> utils2p.synchronization.edges(binary_line)
    (array([1, 4]),)
    >>> utils2p.synchronization.edges(binary_line, size=2)
    (array([], dtype=int64),)
    >>> utils2p.synchronization.edges(binary_line, size=(-np.inf, np.inf))
    (array([1, 3, 4]),)

    >>> continuous_line = np.array([0, 0, 3, 3, 3, 5, 5, 8, 8, 10, 10, 10])
    >>> utils2p.synchronization.edges(continuous_line)
    (array([2, 5, 7, 9]),)
    >>> utils2p.synchronization.edges(continuous_line, size=2)
    (array([2, 7]),)
    >>> utils2p.synchronization.edges(continuous_line, size=(-np.inf, 3))
    (array([5, 9]),)
    """
    if correct_possible_split_edges:
        line = _correct_split_edges(line)
    diff = np.diff(line.astype(np.float64))
    if isinstance(size, tuple):
        zero_elements = np.isclose(diff, np.zeros_like(diff))
        edges_in_range = np.logical_and(diff > size[0], diff < size[1])
        valid_edges = np.logical_and(edges_in_range,
                                     np.logical_not(zero_elements))
        indices = np.where(valid_edges)
    else:
        indices = np.where(diff > size)
    indices = tuple(i + 1 for i in indices)
    return indices


def _correct_split_edges(line):
    """
    This function corrects edges that are spread over multiple ticks.

    Parameters
    ----------
    line : numpy array
        The line for which the edges should be corrected.

    Returns
    -------
    line : numpy array
        Line with corrected edges.

    Examples
    --------
    >>> line = np.array([0, 0, 0, 1, 2, 3, 3, 3, 2, 1, 0, 0, 0])
    >>> utils2p.synchronization.correct_split_edges(line)
    array([0, 0, 0, 3, 3, 3, 3, 3, 3, 3, 0, 0, 0])
    """
    rising_edges = np.where(np.diff(line) > 0)[0] + 1
    falling_edges = np.where(np.diff(line) < 0)[0]

    split_rising_edges = np.where(np.diff(rising_edges) == 1)[0]
    split_falling_edges = np.where(np.diff(falling_edges) == 1)[0]

    if len(split_rising_edges) == 0 and len(split_falling_edges) == 0:
        return line

    first_halfs_rising = rising_edges[split_rising_edges]
    second_halfs_rising = rising_edges[split_rising_edges + 1]
    line[first_halfs_rising] = line[second_halfs_rising]

    first_halfs_falling = falling_edges[split_falling_edges]
    second_halfs_falling = falling_edges[split_falling_edges + 1]
    line[second_halfs_falling] = line[first_halfs_falling]

    # Recursive to get edges spread over more than two ticks
    return _correct_split_edges(line)


def get_start_times(line, times, zero_based_counter=False):
    """
    Get the start times of a digital signal, i.e. the times of the 
    rising edges.
    If the line is a zero based counter, such as the processed 
    `frame_counter` or the processed `cam_line`, there is a possibility 
    that the first element in line is already zero. This corresponds to 
    the case where the acquisition of the first frame was triggered 
    before ThorSync started.
    If `zero_based_counter` is `False` this frame will be dropped, i.e. 
    no time for the frame is returned, since there is no rising edge 
    corresponding to the frame.

    Parameters
    ----------
    line : numpy array
        Line signal from h5 file.
    times : numpy array
        Times returned by :func:`get_times`
    zero_based_counter : boolean
        Indicates whether the line is a zero based counter.

    Returns
    -------
    time_points : list
        List of the start times.

    Examples
    --------
    >>> binary_line = np.array([0, 1, 1, 0, 1, 1])
    >>> times = get_times(len(binary_line), freq=20)
    >>> times
    array([0.  , 0.05, 0.1 , 0.15, 0.2 , 0.25])
    >>> utils2p.synchronization.get_start_times(binary_line, times)
    array([0.05, 0.2 ])
    """
    indices = edges(line, size=(0, np.inf))
    if zero_based_counter and line[0] >= 0:
        if line[0] > 0:
            warnings.warn(f"The counter start with value {line[0]}")
        indices_with_first_frame = np.zeros(len(indices[0]) + 1, dtype=int)
        indices_with_first_frame[1:] = indices[0]
        indices = (indices_with_first_frame, )
    time_points = times[indices]
    return time_points


def _capture_metadata(n_frames, dropped_frames=None):
    """
    Returns a dictionary as it is usually saved by the seven
    camera setup in the "capture_metadata.json" file.
    It assumes that no frames where dropped.

    Parameters
    ----------
    n_frames : list of integers
        Number of frames for each camera.
    dropped_frames : list of list of integers
        Frames that were dropped for each camera.
        Default is None which means no frames where
        dropped.

    Returns
    -------
    capture_info : dict
        Default metadata dictionary for the seven camera
        system.
    """
    if dropped_frames is None:
        dropped_frames = [[] for i in range(len(n_frames))]
    capture_info = {"Frame Counts": {}}
    for cam_idx, n in enumerate(n_frames):
        frames_dict = {}
        current_frame = 0
        for i in range(n):
            while current_frame in dropped_frames[cam_idx]:
                current_frame += 1
            frames_dict[str(i)] = current_frame
            current_frame += 1
        capture_info["Frame Counts"][str(cam_idx)] = frames_dict
    return capture_info


def process_cam_line(line, seven_camera_metadata):
    """
    Removes superfluous signals and uses frame numbers in array.
    The cam line signal form the h5 file is a binary sequence.
    Rising edges mark the acquisition of a new frame.
    The setup keeps producing rising edges after the acquisition of the
    last frame. These rising edges are ignored.
    This function converts the binary line to frame numbers using the
    information stored in the metadata file of the seven camera setup.
    In the metadata file the keys are the indices of the file names
    and the values are the grabbed frame numbers. Suppose the 3
    frame was dropped. Then the entries in the dictionary will
    be as follows:
    "2": 2, "3": 4, "4": 5

    Parameters
    ----------
    line : numpy array
        Line signal from h5 file.
    seven_camera_metadata : string
        Path to the json file saved by our camera software.
        This file is usually located in the same folder as the frames
        and is called 'capture_metadata.json'. If None, it is assumed
        that no frames were dropped.

    Returns
    -------
    processed_line : numpy array
        Array with frame number for each time point.
        If no frame is available for a given time,
        the value is -9223372036854775808.
    """
    # Check that sequence is binary
    if len(set(line)) > 2:
        raise ValueError("Invalid line argument. Sequence is not binary.")

    # Find indices of the start of each frame acquisition
    rising_edges = edges(line, (0, np.inf))[0]

    # Load capture metadata or generate default
    if seven_camera_metadata is not None:
        with open(seven_camera_metadata, "r") as f:
            capture_info = json.load(f)
    else:
        capture_info = _capture_metadata([
            len(rising_edges),
        ])

    # Find the number of frames for each camera
    n_frames = []
    for cam_idx in capture_info["Frame Counts"].keys():
        max_in_json = max(capture_info["Frame Counts"][cam_idx].values())
        n_frames.append(max_in_json + 1)

    # Ensure all cameras acquired the same number of frames
    if len(np.unique(n_frames)) > 1:
        raise SynchronizationError(
            "The frames across cameras are not synchronized.")

    # Last rising edge that corresponds to a frame
    last_tick = max(n_frames)

    # check that there is a rising edge for every frame
    if len(rising_edges) < last_tick:
        raise ValueError(
            "The provided cam line and metadata are inconsistent. " +
            "cam line has less frame acquisitions than metadata.")

    # Ensure correct handling if no rising edges are present after last frame
    if len(rising_edges) == int(last_tick):
        average_frame_length = int(np.mean(np.diff(rising_edges)))
        last_rising_edge = rising_edges[-1]
        additional_edge = last_rising_edge + average_frame_length
        if additional_edge > len(line):
            additional_edge = len(line)
        rising_edges = list(rising_edges)
        rising_edges.append(additional_edge)
        rising_edges = np.array(rising_edges)

    processed_line = np.ones_like(line) * np.nan

    current_frame = 0
    first_camera_used = sorted(list(capture_info["Frame Counts"].keys()))[0]
    for i, (start, stop) in enumerate(
            zip(rising_edges[:last_tick], rising_edges[1:last_tick + 1])):
        if capture_info["Frame Counts"][first_camera_used][str(current_frame +
                                                               1)] <= i:
            current_frame += 1
        processed_line[start:stop] = current_frame
    return processed_line


def process_frame_counter(line, metadata=None, steps_per_frame=None):
    """
    Converts the frame counter line to an array with frame numbers for each
    time point.

    Parameters
    ----------
    line : numpy array
        Line signal from h5 file.
    metadata : :class:`Metadata`
        :class:`Metadata` object that holding the 2p imaging
        metadata for the experiment. Optional. If metadata is not
        given steps_per_frame has to be set.
    steps_per_frame : int
        Number of steps the frame counter takes per frame.
        This includes fly back frames and averaging, i.e. if you
        acquire one frame and flyback frames is set to 3 this number
        should be 4.

    Returns
    -------
    processed_frame_counter : numpy array
        Array with frame number for each time point.
        If no frame was recorded at a time point, 
        the value is -9223372036854775808.
    """
    if metadata is not None and steps_per_frame is not None:
        warnings.warn("metadata argument will be ignored " +
                      "because steps_per_frame argument was set.")
    if metadata is not None and not isinstance(metadata, utils2p.Metadata):
        raise TypeError(
            "metadata argument must be of type utils2p.Metadata or None.")
    if steps_per_frame is not None and not isinstance(steps_per_frame, int):
        raise TypeError(f"steps_per_frame has to be of type int not {type(steps_per_frame)}")

    if metadata is not None and steps_per_frame is None:
        if metadata.get_value("Streaming", "zFastEnable") == "0":
            steps_per_frame = 1
        else:
            steps_per_frame = metadata.get_n_z()
            if metadata.get_value("Streaming", "enable") == "1":
                steps_per_frame += metadata.get_n_flyback_frames()
        if metadata.get_value(
                "LSM",
                "averageMode") == "1" and metadata.get_area_mode() not in [
                    "line", "kymograph"
                ]:
            steps_per_frame = steps_per_frame * metadata.get_n_averaging()
    elif steps_per_frame is None:
        raise ValueError("If no metadata object is given, " +
                         "the steps_per_frame argument has to be set.")

    processed_frame_counter = np.ones_like(line) * np.nan
    rising_edges = edges(line, (0, np.inf))[0]

    # Case of one frame/volume only
    if len(rising_edges) <= steps_per_frame:
        processed_frame_counter[rising_edges[0]:] = 0
        return processed_frame_counter

    for i, index in enumerate(
            range(0,
                  len(rising_edges) - steps_per_frame, steps_per_frame)):
        processed_frame_counter[
            rising_edges[index]:rising_edges[index + steps_per_frame]] = i
    processed_frame_counter[rising_edges[-1 * steps_per_frame]:] = (
        processed_frame_counter[rising_edges[-1 * steps_per_frame] - 1] + 1)
    return processed_frame_counter


def process_stimulus_line(line):
    """
    This function converts the stimulus line to an array with
    0s and 1s for stimulus off and on respectively. The raw
    stimulus line can contain values larger than 1.

    Parameters
    ----------
    line : numpy array
        Line signal from h5 file.

    Returns
    -------
    processed_frame_counter : numpy array
        Array with binary stimulus state for each time point.
    """
    processed_stimulus_line = np.zeros_like(line)
    indices = np.where(line > 0)
    processed_stimulus_line[indices] = 1
    return processed_stimulus_line


def _crop_lines(mask, lines):
    """
    This function crops all lines based on a binary signal/mask.
    The 'Capture On' line of the h5 file can be used as a mask.

    Parameters
    ----------
    mask : numpy array
        Mask that is used for cropping.
    lines : list of numpy arrays
        List of the lines that should be cropped.

    Returns
    -------
    cropped_lines : tuple of numpy arrays
        Tuple of cropped lines in same order as in input list.
    """
    indices = np.where(mask)[0]
    first_idx = indices[0]
    last_idx = indices[-1]
    cropped_lines = []
    for line in lines:
        cropped_lines.append(line[first_idx:last_idx + 1])
    return tuple(cropped_lines)


def beh_idx_to_2p_idx(beh_indices, cam_line, frame_counter):
    """
    This functions converts behaviour frame numbers into the corresponding
    2p frame numbers.

    Parameters
    ----------
    beh_indices : numpy array
        Indices of the behaviour frames to be converted.
    cam_line : numpy array
        Processed cam line.
    frame_counter : numpy array
        Processed frame counter.

    Returns
    -------
    indices_2p : numpy array
        Corresponding 2p frame indices.
    """
    thor_sync_indices = edges(cam_line)[0]
    if not cam_line[0] < 0:
        thor_sync_indices = np.append(np.array([0]), thor_sync_indices)

    indices_2p = np.ones(len(beh_indices), dtype=int) * np.nan

    first_frame_of_cam_line = np.min(cam_line[np.where(cam_line >= 0)])

    for i, frame_num in enumerate(beh_indices):

        # This is necessary for cropped lines that don't start at 0
        frame_num = frame_num - first_frame_of_cam_line
        if frame_num < 0:
            raise ValueError(f"{frame_num + first_frame_of_cam_line} is smaller than first frame in cam_line ({first_frame_of_cam_line})")

        thor_sync_index = thor_sync_indices[frame_num]
        indices_2p[i] = frame_counter[thor_sync_index]

    return indices_2p


class SyncMetadata(utils2p._XMLFile):
    """
    Class for managing ThorSync metadata.
    Loads metadata file 'ThorRealTimeDataSettings.xml'
    and returns the root of an ElementTree.

    Parameters
    ----------
    path : string
        Path to xml file.

    Returns
    -------
    Instance of class Metadata
        Based on given xml file.
    """
    def get_active_devices(self):
        active_devices = []
        for device in self.get_value("DaqDevices", "AcquireBoard"):
            if device.attrib["active"] == "1":
                active_devices.append(device)
        return active_devices

    def get_freq(self):
        """
        Returns the frequency of the ThorSync
        value acquisition, i.e. the sample rate.

        Returns
        -------
        freq : integer
            Sample frequency in Hz.

        Examples
        --------
        >>> import utils2p.synchronization
        >>> metadata = utils2p.synchronization.SyncMetadata("data/mouse_kidney_raw/2p/Sync-025/ThorRealTimeDataSettings.xml")
        >>> metadata.get_freq()
        30000
        """
        sample_rate = -1
        for device in self.get_active_devices():
            set_for_device = False
            for element in device.findall("SampleRate"):
                if element.attrib["enable"] == "1":
                    if set_for_device:
                        raise ValueError(
                            "Invalid metadata file. Multiple sample rates " +
                            f"are enabled for device {device.type}")
                    if sample_rate != -1:
                        raise ValueError("Multiple devices are enabled.")
                    sample_rate = int(element.attrib["rate"])
                    set_for_device = True
        return sample_rate


def get_processed_lines(sync_file,
                        sync_metadata_file,
                        seven_camera_metadata_file,
                        metadata_2p_file=None,
                        ):
    """
    This function extracts all the standard lines and processes them.
    It works for both 2p microscopes, and for old/new thorsync versions.
    If metadata_2p file is None, it ignores processing of imaging lines.

    Parameters
    ----------
    sync_file : str
        Path to the synchronization file.
    sync_metadata_file : str
        Path to the synchronization metadata file.
    seven_camera_metadata_file : str
        Path to the metadata file of the 7 camera system.
    metadata_2p_file : str, optional
        Path to the ThorImage metadata file (2p imaging).
    
    Returns
    -------
    processed_lines : dictionary
        Dictionary with all processed lines.
    """
    
    # get lines from sync file
    lines = _get_appropriate_lines(sync_file)
    processed_lines = lines.copy()

    # process behavior lines
    processed_lines["Cameras"] = process_cam_line(processed_lines["Cameras"], seven_camera_metadata_file)

    # if imaging present, process lines
    if metadata_2p_file is not None:
        metadata_2p = utils2p.Metadata(metadata_2p_file)
        processed_lines["FrameCounter"] = process_frame_counter(
            processed_lines["FrameCounter"], metadata_2p)

    # process opto line
    processed_lines["opto_stim"] = process_stimulus_line(processed_lines["opto_stim"])

    # Make sure the clipping start just before the
    # acquisition of the first frame
    # (only if there is imaging)
    if metadata_2p_file is not None:
        mask = np.logical_and(processed_lines["CaptureOn"],
                            processed_lines["FrameCounter"] >= 0)
        indices = np.where(mask)[0]
        mask[max(0, indices[0] - 1)] = True

        for line_name, _ in processed_lines.items():
            processed_lines[line_name] = _crop_lines(mask, [
                processed_lines[line_name],
            ])[0]

    # Get times of ThorSync ticks
    metadata = SyncMetadata(sync_metadata_file)
    freq = metadata.get_freq()
    times = get_times(len(processed_lines["FrameCounter"]), freq)
    processed_lines["Times"] = times

    return processed_lines

def _get_appropriate_lines(sync_file):
    """ Function to get the relevant lines from the sync file,
    depending on the version of ThorSync and the microscope used.
    
    Parameters
    ----------
    sync_file : str
        Path to the synchronization file.
    
    Returns
    -------
    processed_lines : dict
        Dictionary with all relevant lines.
    """
    # initialize dictionary
    processed_lines = {}

    # get scope from user_config
    scope = user_config['scope']
    # warning for scope 1 - NOT TESTED!!! (remove this if it works as expected)
    if scope == '2p_1':
        print('WARNING: SCOPE 1 SYNC FILE READING HAS NOT BEEN TESTED!! MAKE SURE IT WORKS')

    # get line names that depend on scope - and their location within the sync file
    if scope == '2p_1':
        lines_scope = {"opto_stim": ["DI","CO2_Stim"], "Cameras":["DI","Basler"]}
    elif scope == '2p_2':
        lines_scope = {"opto_stim": ["DI","LightSheetLaserOn"], "Cameras":["DI","Cameras"]}
    # load lines
    with h5py.File(sync_file, "r") as f:
        for name, (line_type, line_name) in lines_scope.items():
            processed_lines[name] = f[line_type][line_name][:].squeeze()

    # try loading other lines - if fail, use old version of line names (Thorsync version)
    lines = {"CaptureOn": ["DI","CaptureOn"], "FrameCounter":["CI","FrameCounter"]}
    with h5py.File(sync_file, "r") as f:
        try:
            for name, (line_type, line_name) in lines.items():
                processed_lines[name] = f[line_type][line_name][:].squeeze()
        except:
            lines = {"CaptureOn": ["DI","Capture On"], "FrameCounter":["CI","Frame Counter"]}
            for name, (line_type, line_name) in lines.items():
                processed_lines[name] = f[line_type][line_name][:].squeeze()

    return processed_lines
