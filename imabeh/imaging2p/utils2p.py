"""
General utils functions to work with 2p imaging data.

COPIED FROM NeLy-EPFL/utils2p (almost identical) - changes:
- updated to newer version of tifffile
- edited create_tiffs to work with either 1 or 2 channels


Available functions:

* FUNCTIONS TO FIND IMAGING SPECIFIC FILES
    - find_file
    - find_raw_file
    - find_metadata_file

* FUNCTIONS TO MANAGE METADATA
    - Metadata (class)
    - get_metadata_value
    - get_n_time_points, get_num_x_pixels, get_num_y_pixels, get_area_mode
    - get_n_z, get_n_averaging, get_n_channels,get_channels
    - get_pixel_size, get_z_step_size, get_z_pixel_size, get_width
    - get_dwell_time, get_n_flyback_frames, get_frame_rate
    - get_power_reg1_start, get_gain_a, get_gain_b, get_date_time

* FUNCTIONS TO LOAD AND SAVE DATA
    - load_img, load_raw, save_img
    - load_stack_batches, load_stack_patches
    - load_z_stack, concatenate_z

* FUNCTIONS TO PROCESS DATA (most useful)
    - create_tiffs
"""

import os
import numpy as np
import math
import tifffile
import array
import glob
import xml.etree.ElementTree as ET
from pathlib import Path


# FUNCTIONS TO MANAGE METADATA

def _node_crawler(node, *args):
    """
    This function is a helper function for Metadata.get_metadata_value.
    It crawls through the xml tree to find the desired value.
    """
    if len(args) == 0:
        return node
    elif len(args) == 1 and args[0] in node.attrib.keys():
        return node.attrib[args[0]]
    if len(node) == 0:
        raise ValueError(f"Hit dead end {node} has no children.")
    return [_node_crawler(child, *args[1:]) for child in node.findall(args[0])]

class _XMLFile:
    """
    Base class for xml based Metadata.
    """
    def __init__(self, path):
        self.path = path
        self.tree = ET.parse(path)
        self.root = self.tree.getroot()

    def get_value(self, *args):
        node = self.root.find(args[0])
        values = _node_crawler(node, *args[1:])
        if len(values) == 1:
            return values[0]
        return values

class Metadata(_XMLFile):
    """
    Class for managing ThorImage metadata.
    Loads metadata file 'Experiment.xml' and returns the root of an ElementTree.

    Parameters
    ----------
    path : string
        Path to xml file.

    Returns
    -------
    Instance of class Metadata
        Based on given xml file.

    Example
    --------
    >>> from imabeh.imaging import imaging
    >>> metadata = imaging.Metadata("data/mouse_kidney_z_stack/Experiment.xml")
    >>> type(metadata)
    <class 'utils2p.main.Metadata'>
    """

    def __repr__(self):
        # self.root.getchildren() will list all the datatypes, but here we just
        # show ones containing data that is most often useful.
        datatypes = ['LSM', 'Timelapse', 'ZStage', 'Wavelengths', 'Streaming',
                      'PowerRegulator', 'PMT', 'Date']
        return ('<' +
                ',\n\n'.join(['{}: {}'.format(x, self.root.find(x).attrib)
                              for x in datatypes])
                + '>')

    def get_metadata_value(self, *args):
        """
        This function returns a value from the metadata file 'Experiment.xml'.

        Parameters
        ----------
        args : strings
            Arbitrary number of strings of tags from the xml file in the
            correct order. See examples.

        Returns
        -------
        attribute or node : string or ElementTree node
            If the number of strings given in args leads to a leaf of the tree,
            the attribute, usually a dictionary, is returned.
            Otherwise the node is returned.

        Examples
        --------
        >>> from imabeh.imaging import imaging
        >>> metadata = imaging.Metadata('data/mouse_kidney_time_series_z_stack/Experiment.xml')
        >>> metadata.get_metadata_value('Timelapse','timepoints')
        '3'
        """
        return self.get_value(*args)

    def get_n_time_points(self):
        """
        Returns the number of time points for a given experiment metadata.

        Returns
        -------
        n_time_points : int
            Number of time points.

        Examples
        --------
        >>> from imabeh.imaging import imaging
        >>> metadata = imaging.Metadata('data/mouse_kidney_time_series_z_stack/Experiment.xml')
        >>> metadata.get_n_time_points()
        3000
        """
        return int(self.get_metadata_value("Timelapse", "timepoints"))

    def get_num_x_pixels(self):
        """
        Returns the image width for a given experiment metadata,
        i.e. the number of pixels in the x direction.

        Returns
        -------
        width : int
            Width of image.
        """
        return int(self.get_metadata_value("LSM", "pixelX"))

    def get_num_y_pixels(self):
        """
        Returns the image height for a given experiment metadata,
        i.e. the number of pixels in the y direction.

        Returns
        -------
        height : int
            Width of image.
        """
        return int(self.get_metadata_value("LSM", "pixelY"))

    def get_area_mode(self):
        """
        Returns the area mode of a given experiment metadata, e.g.
        square, rectangle, line, kymograph.

        Returns
        -------
        area_mode : string
            Area mode of experiment.
        """
        int_area_mode = int(self.get_metadata_value("LSM", "areaMode"))
        if int_area_mode == 0:
            return "square"
        if int_area_mode == 1:
            return "rectangle"
        if int_area_mode == 2:
            return "kymograph"
        if int_area_mode == 3:
            return "line"
        raise ValueError(f"{int_area_mode} is not a valid value for areaMode.")

    def get_n_z(self):
        """
        Returns the number for z slices for a given experiment metadata.

        Returns
        -------
        n_z : int
            Number of z layers of image.
        """
        return int(self.get_metadata_value("ZStage", "steps"))

    def get_n_averaging(self):
        """
        Returns the number of frames that are averaged.

        Returns
        -------
        n_averaging : int
            Number of averaged frames.
        """
        return int(self.get_value("LSM", "averageNum"))

    def get_n_channels(self):
        """
        Returns the number of channels for a given experiment metadata.

        Returns
        -------
        n_channels : int
            Number of channels in raw data file.
        """
        return len(self.get_metadata_value("Wavelengths")) - 1

    def get_channels(self):
        """
        Returns a tuple with the names of all channels.

        Returns
        -------
        channels : tuple of strings
            Names of channels.
        """
        channels = self.get_metadata_value("Wavelengths", "Wavelength", "name")
        return tuple(channels)

    def get_pixel_size(self):
        """
        Returns the pixel size for a given experiment metadata.

        Returns
        -------
        pixel_size : float
            Size of one pixel in um in x and y direction.
        """
        return float(self.get_metadata_value("LSM", "pixelSizeUM"))

    def get_z_step_size(self):
        """
        Returns the z step size for a given experiment metadata.

        Returns
        -------
        z_step_size : float
            Distance covered in um along z direction.
        """
        return float(self.get_metadata_value("ZStage", "stepSizeUM"))

    def get_z_pixel_size(self):
        """
        Returns the pixel size in z direction for a given experiment metadata.
        This function is meant for "kymograph" and "line" recordings.
        For these recordings the pixel size in z direction is not equal to the step size, 
        unless the number of pixels equals the number of steps.
        For all other types of recordings it is equivalent to :func:`get_z_step_size`.

        Returns
        -------
        z_pixel_size : float
            Distance covered in um along z direction.
        """
        area_mode = self.get_area_mode()
        if area_mode in ('line', 'kymograph'):
            return (float(self.get_metadata_value("ZStage", "stepSizeUM")) *
                    self.get_n_z() / self.get_num_y_pixels())
        return float(self.get_metadata_value("ZStage", "stepSizeUM"))

    def get_dwell_time(self):
        """
        Returns the dwell time for a given experiment metadata.

        Returns
        -------
        dwell_time : float
            Dwell time for a pixel.
        """
        return float(self.get_metadata_value("LSM", "dwellTime"))

    def get_n_flyback_frames(self):
        """
        Returns the number of flyback frames.

        Returns
        -------
        n_flyback : int
            Number of flyback frames.
        """
        n_flyback = int(self.get_metadata_value("Streaming", "flybackFrames"))
        return n_flyback

    def get_frame_rate(self):
        """
        Returns the frame rate for a given experiment metadata.
        When the frame rate is calculated flyback frames and steps in z are not considered frames.

        Returns
        -------
        frame_rate : float
            Frame rate of the experiment.
        """
        frame_rate_without_flybacks = float(
            self.get_metadata_value("LSM", "frameRate"))
        flyback_frames = self.get_n_flyback_frames()
        number_of_slices = self.get_n_z()
        return frame_rate_without_flybacks / (flyback_frames + number_of_slices)
    
    def get_width(self):
        """
        Returns the image width in um for a given experiment metadata.

        Returns
        -------
        width : float
            Width of FOV in um.
        """
        return float(self.get_metadata_value("LSM", "widthUM"))

    def get_power_reg1_start(self):
        """
        Returns the starting position of power regulator 1 for a given experiment metadata.
        Unless a gradient is defined, this value is the power value for the entire experiment.

        Returns
        -------
        reg1_start : float
            Starting position of power regulator 1.
        """
        return float(self.get_metadata_value("PowerRegulator", "start"))

    def get_gain_a(self):
        """
        Returns the gain of channel A for a given experiment metadata.

        Returns
        -------
        gainA : int
            Gain of channel A.
        """
        return float(self.get_metadata_value("PMT", "gainA"))

    def get_gain_b(self):
        """
        Returns the gain of channel B for a given experiment metadata.

        Returns
        -------
        gainB : int
            Gain of channel B.
        """
        return float(self.get_metadata_value("PMT", "gainB"))

    def get_date_time(self):
        """
        Returns the date and time of an experiment for a given experiment metadata.

        Returns
        -------
        date_time : string
            Date and time of an experiment.
        """
        return self.get_metadata_value("Date", "date")



# FUNCTIONS TO FIND FILES

def _find_file(directory, name, file_type, most_recent=False):
    """
    This function finds a unique file with a given name in the directory.
    If multiple files with this name are found and most_recent = False, it throws an exception.
    otherwise, it returns the most recent file.

    Parameters
    ----------
    directory : str
        Directory in which to search.
    name : str
        Name of the file.
    file_type : str
        Type of the file (for reporting errors only)
    most_recent : bool, optional
        If True, return the most recent file if multiple files are found, by default False

    Returns
    -------
    path : str
        Path to file.
    """
    file_names = list(Path(directory).rglob(name))
    if len(file_names) > 1 and not most_recent:
        raise RuntimeError(
            f"Could not identify {file_type} file unambiguously. " +
            f"Discovered {len(file_names)} {file_type} files in {directory}."
        )
    elif len(file_names) > 1 and most_recent:
        file_names = sorted(file_names, key=lambda x: x.stat().st_mtime, reverse=True)
    elif len(file_names) == 0:
        raise FileNotFoundError(f"No {file_type} file found in {directory}")
    
    return str(file_names[0])

def find_raw_file(directory):
    """
    This function finds the path to the raw file
    "Image_0001_0001.raw" created by ThorImage and returns it.
    If multiple files with this name are found, it throws
    an exception unless.

    Parameters
    ----------
    directory : str
        Directory in which to search.

    Returns
    -------
    path : str
        Path to raw file.
    """
    # some versions of ThorImage save the raw file as "Image_001_001.raw or Image_0001_0001.raw"
    try:
        return _find_file(directory,
                          "Image_0001_0001.raw",
                          "raw")
    except:
        return _find_file(directory,
                        "Image_001_001.raw",
                        "raw")

def find_metadata_file(directory):
    """
    This function finds the path to the metadata file
    "Experiment.xml" created by ThorImage and returns it.
    If multiple files with this name are found, it throws
    an exception.

    Parameters
    ----------
    directory : str
        Directory in which to search.

    Returns
    -------
    path : str
        Path to metadata file.
    """
    return _find_file(directory,
                      "Experiment.xml",
                      "metadata")


# FUNCTIONS TO LOAD AND SAVE DATA

def load_img(path, memmap=False):
    """
    This functions loads an image from file and returns as a numpy array.

    Parameters
    ----------
    path : string
        Path to image file.
    memmap : bool
        If `True`, the image is not loaded into memory but remains on
        disk and a `numpy.memmap` object is returned. It can be indexed
        like a normal `numpy.array`. This option is useful when a stack
        does not fit into memory. It can also be used when opening many
        stacks simultaneously.

    Returns
    -------
    numpy.array or numpy.memmap
        Image in form of numpy array or numpy memmap.

    Examples
    --------
    >>> import utils2p
    >>> img = utils2p.load_img("data/chessboard_GRAY_U16.tif")
    >>> type(img)
    <class 'numpy.ndarray'>
    >>> img.shape
    (200, 200)
    >>> img = utils2p.load_img("data/chessboard_GRAY_U16.tif", memmap=True)
    >>> type(img)
    <class 'numpy.memmap'>
    """
    path = os.path.expanduser(os.path.expandvars(path))
    if memmap:
        return tifffile.memmap(path)
    else:
        return tifffile.imread(path)

def load_stack_batches(path, batch_size):
    """
    This function loads a stack in several batches to make sure
    the system does not run out of memory. It returns a generator
    that yields consecutive chunks of `batch_size` frames of the stack.
    The remaining memory is freed up by the function until the generator
    is called again.

    Parameters
    ----------
    path : string
        Path to stack.
    batch_size : int
        Number of frames in one chunk.

    Returns
    -------
    generator
        Generator that yields chunks of `batch_size` frames of the
        stack.
    """
    stack = load_img(path, memmap=True)
    if stack.ndim < 3:
        raise ValueError(
            f"The path does not point to a stack. The shape is {stack.shape}.")
    n_batches = int(stack.shape[0] / batch_size) + 1
    for i in range(n_batches):
        substack = np.array(stack[i * batch_size:(i + 1) * batch_size])
        yield substack

def load_stack_patches(path, patch_size, padding=0, return_indices=False):
    """
    Returns a generator that yields patches of the stack of images.
    This is useful when multiple stacks should be processed but they
    don't fit into memory, e.g. when computing an overall fluorescence
    baseline for all trials of a fly.

    Parameters
    ----------
    path : string
       Path to stack.
    patch_size : tuple of two integers
       Size of the patch returned.
    padding : integer or tuple of two integers
       The amount of overlap between patches. Note that this increases
       the effective patch size. Default is 0. If tuple, different padding
       is used for the dimensions.
    return_indices : boolean
       If True, the indices necessary for slicing to generate the patch and
       the indices necessary for slicing to remove the padding from the
       returned patch are returned. Default is False.
       The values are retuned in the following form:
       ```
       indices = [[start_patch_dim_0, stop_patch_dim_0],
                 [start_patch_dim_1, stop_patch_dim_1],]
       patch_indices = [[start_after_padding_dim_0, stop_after_padding_dim_0],
                        [start_after_padding_dim_1, stop_after_padding_dim_1],]
       ```

    Returns
    -------
    patch : numpy array
        Patch of the stack.
    indices : tuples of integers, optional
        See description of the `return_indices` parameter above
        and examples below.
    patch_indices : tuples of integers, optinal
        See description of the `return_indices` parameter above
        and examples below.

    Examples
    --------
    >>> import numpy as np
    >>> import utils2p
    >>> metadata = utils2p.Metadata('data/mouse_kidney_raw/2p/Untitled_001/Experiment.xml')
    >>> stack1, stack2 = utils2p.load_raw('data/mouse_kidney_raw/2p/Untitled_001/Image_0001_0001.raw',metadata)
    >>> print(stack1.shape)
    (5, 256, 256)
    >>> utils2p.save_img('stack1.tif',stack1)
    >>> generator = utils2p.load_stack_patches('stack1.tif', (5, 4))
    >>> first_patch = next(generator)
    >>> print(first_patch.shape)
    (5, 5, 4)
    >>> generator = utils2p.load_stack_patches('stack1.tif', (15, 20), padding=3, return_indices=True)
    >>> first_patch, indices, patch_indices = next(generator)
    >>> print(first_patch.shape)
    (5, 18, 23)
    >>> print(patch_indices)
    [[0, 15], [0, 20]]
    >>> print(indices)
    [[0, 15], [0, 20]]
    >>> first_patch_without_padding = first_patch[:, patch_indices[0][0] : patch_indices[0][1], patch_indices[1][0] : patch_indices[1][1]]
    >>> print(first_patch_without_padding.shape)
    (5, 15, 20)
    >>> np.all(stack1[:, indices[0][0] : indices[0][1], indices[1][0] : indices[1][1]] == first_patch_without_padding)
    True

    Note that the patch has no padding at the edges.
    When looking at the second patch we see that it is padded on both side
    in the second dimension but still only on one side of the first dimension.

    >>> second_patch, indices, patch_indices = next(generator)
    >>> print(second_patch.shape)
    (5, 18, 26)
    >>> print(patch_indices)
    [[0, 15], [3, 23]]
    >>> print(indices)
    [[0, 15], [20, 40]]
    >>> second_patch_without_padding = second_patch[:, patch_indices[0][0] : patch_indices[0][1], patch_indices[1][0] : patch_indices[1][1]]
    >>> print(second_patch_without_padding.shape)
    (5, 15, 20)
    """
    stack = load_img(path, memmap=True)
    dims = stack.shape[1:]
    n_patches_0 = math.ceil(dims[0] / patch_size[0])
    n_patches_1 = math.ceil(dims[1] / patch_size[1])
    if isinstance(padding, int):
        padding = (padding, padding)
    for i in range(n_patches_0):
        for j in range(n_patches_1):
            indices = [
                [patch_size[0] * i, patch_size[0] * (i + 1)],
                [patch_size[1] * j, patch_size[1] * (j + 1)],
            ]
            start_dim_0 = max(indices[0][0] - padding[0], 0)
            start_dim_1 = max(indices[1][0] - padding[1], 0)
            stop_dim_0 = min(indices[0][1] + padding[0], dims[0])
            stop_dim_1 = min(indices[1][1] + padding[1], dims[1])
            patch = stack[:, start_dim_0:stop_dim_0,
                          start_dim_1:stop_dim_1].copy()
            del stack
            if not return_indices:
                yield patch
            else:
                offset_dim_0 = indices[0][0] - start_dim_0
                offset_dim_1 = indices[1][0] - start_dim_1
                patch_indices = [
                    [offset_dim_0, patch_size[0] + offset_dim_0],
                    [offset_dim_1, patch_size[1] + offset_dim_1],
                ]
                yield patch, indices, patch_indices
            stack = load_img(path)

def load_raw(path, metadata):
    """
    This function loads a raw image generated by ThorImage as a numpy array.

    Parameters
    ----------
    path : string
        Path to raw file.
    metadata : ElementTree root
        Can be obtained with :func:`get_metadata`.

    Returns
    -------
    stacks : tuple of numpy arrays
        Number of numpy arrays depends on the number of channels recoded during
        the experiment. Has the following dimensions:
        TZYX or TYX for planar images.

    Examples
    --------
    >>> import utils2p
    >>> metadata = utils2p.Metadata('data/mouse_kidney_raw/2p/Untitled_001/Experiment.xml')
    >>> stack1, stack2 = utils2p.load_raw('data/mouse_kidney_raw/2p/Untitled_001/Image_0001_0001.raw',metadata)
    >>> type(stack1), type(stack2)
    (<class 'numpy.ndarray'>, <class 'numpy.ndarray'>)
    >>> utils2p.save_img('stack1.tif',stack1)
    >>> utils2p.save_img('stack2.tif',stack2)
    """
    path = os.path.expanduser(os.path.expandvars(path))
    n_time_points = metadata.get_n_time_points()
    width = metadata.get_num_x_pixels()
    height = metadata.get_num_y_pixels()
    n_channels = metadata.get_n_channels()
    byte_size = os.stat(path).st_size

    assert not byte_size % 1, "File does not have an integer byte length."
    byte_size = int(byte_size)

    n_z = (
        byte_size / 2 / width / height / n_time_points / n_channels
    )  # divide by two because the values are of type short (16bit = 2byte)

    assert (
        not n_z %
        1), "Size given in metadata does not match the size of the raw file."
    n_z = int(n_z)

    # number of z slices from meta data can be different
    # because of flyback frames
    meta_n_z = metadata.get_n_z()

    if n_z == 1:
        stacks = np.zeros((n_channels, n_time_points, height, width),
                          dtype="uint16")
        image_size = width * height
        # number of values stored for a given time point
        # (this includes images for all channels)
        t_size = (width * height * n_channels)
        with open(path, "rb") as f:
            for t in range(n_time_points):
                # print('{}/{}'.format(t,n_time_points))
                a = array.array("H")
                a.fromfile(f, t_size)
                for c in range(n_channels):
                    stacks[c, t, :, :] = np.array(a[c * image_size:(c + 1) *
                                                    image_size]).reshape(
                                                        (height, width))
    elif n_z > 1:
        stacks = np.zeros((n_channels, n_time_points, meta_n_z, height, width),
                          dtype="uint16")
        image_size = width * height
        t_size = (
            width * height * n_z * n_channels
        )  # number of values stored for a given time point (this includes images for all channels)
        with open(path, "rb") as f:
            for t in range(n_time_points):
                # print('{}/{}'.format(t,n_time_points))
                a = array.array("H")
                a.fromfile(f, t_size)
                a = np.array(a).reshape(
                    (-1, image_size
                     ))  # each row is an image alternating between channels
                for c in range(n_channels):
                    stacks[c, t, :, :, :] = a[c::n_channels, :].reshape(
                        (n_z, height, width))[:meta_n_z, :, :]

    area_mode = metadata.get_area_mode()
    if area_mode in ('line', 'kymograph') and meta_n_z > 1:
        concatenated = []
        for stack in stacks:
            concatenated.append(concatenate_z(stack))
        stacks = concatenated

    if len(stacks) == 1:
        return (np.squeeze(stacks[0]), )
    return tuple(np.squeeze(stacks))

def load_z_stack(path, metadata):
    """
    Loads tif files as saved when capturing a z-stack into a 3D numpy array.

    Parameters
    ----------
    path : string
        Path to directory of the z-stack.
    metadata : ElementTree root
        Can be obtained with :func:`get_metadata`.

    Returns
    -------
    stacks : tuple of numpy arrays
        Z-stacks for Channel A (green) and Channel B (red).

    Examples
    --------
    >>> import utils2p
    >>> metadata = utils2p.Metadata("data/mouse_kidney_z_stack/Experiment.xml")
    >>> z_stack_A, z_stack_B = utils2p.load_z_stack("data/mouse_kidney_z_stack/", metadata)
    >>> type(z_stack_A), type(z_stack_B)
    (<class 'numpy.ndarray'>, <class 'numpy.ndarray'>)
    >>> z_stack_A.shape, z_stack_B.shape
    ((3, 128, 128), (3, 128, 128))
    """
    path = os.path.expanduser(os.path.expandvars(path))
    channels = metadata.get_channels()
    paths = sorted(glob.glob(os.path.join(path, channels[0]) + "*.tif"))
    stacks = load_img(paths[0])
    if stacks.ndim == 5:
        return tuple([stacks[:, :, 0, :, :], stacks[:, :, 1, :, :]])
    return tuple([stacks[:, 0, :, :], stacks[:, 1, :, :]])

def concatenate_z(stack):
    """
    Concatenate in z direction for area mode 'line' or 'kymograph',
    e.g. coronal section. This is necessary because z steps are
    otherwise treated as additional temporal frame, i.e. in Fiji
    the frames jump up and down between z positions.

    Parameters
    ----------
    stack : 4D or 6D numpy array
        Stack to be z concatenated.

    Returns
    -------
    stack : 3D or 5D numpy array
        Concatenated stack.

    Examples
    --------
    >>> import utils2p
    >>> import numpy as np
    >>> stack = np.zeros((100, 2, 64, 128))
    >>> concatenated = utils2p.concatenate_z(stack)
    >>> concatenated.shape
    (100, 128, 128)
    """
    res = np.concatenate(np.split(stack, stack.shape[-3], axis=-3), axis=-2)
    return np.squeeze(res)

def save_img(path,
             img,
             imagej=True,
             color=False,
             full_dynamic_range=True,
             metadata=None):
    """
    Saves an image that is given as a numpy array to file.

    Parameters
    ----------
    path : string
        Path where the file is saved.
    img : numpy array
        Image or stack. For stacks, the first dimension is the stack index.
        For color images, the last dimension are the RGB channels.
    imagej : boolean
        Save imagej compatible stacks and hyperstacks.
    color : boolean, default = False
        Determines if image is RGB or gray scale.
        Will be converted to uint8.
    full_dynamic_range : boolean, default = True
        When an image is converted to uint8 for saving a color image the
        max value of the output image is the max of uint8,
        i.e. the image uses the full dynamic range available.
    """
    if img.dtype == bool:  # used to be np.bool
        img = img.astype(np.uint8) * 255
    path = os.path.expanduser(os.path.expandvars(path))
    if color:
        if img.dtype != np.uint8:
            old_max = np.max(img, axis=tuple(range(img.ndim - 1)))
            if not full_dynamic_range:
                if np.issubdtype(img.dtype, np.integer):
                    old_max = np.iinfo(img.dtype).max * np.ones(3)
                elif np.issubdtype(img.dtype, np.floating):
                    old_max = np.finfo(img.dtype).max * np.ones(3)
                else:
                    raise ValueError(
                        f"img must be integer or float type not {img.dtype}")
            new_max = np.iinfo(np.uint8).max
            img = img / old_max * new_max
            img = img.astype(np.uint8)
        if imagej and img.ndim == 4:
            img = np.expand_dims(img, axis=1)
        if imagej and img.ndim == 3:
            img = np.expand_dims(img, axis=0)
            img = np.expand_dims(img, axis=1)
    else:
        if imagej and img.ndim == 4:
            img = np.expand_dims(img, axis=2)
            img = np.expand_dims(img, axis=5)
        if imagej and img.ndim == 3:
            img = np.expand_dims(img, axis=1)
            img = np.expand_dims(img, axis=4)
    if img.dtype == np.float64:
        img = img.astype(np.float32)
    if metadata is None:
        tifffile.imsave(path, img, imagej=imagej)
    else:
        # TODO add meta data like metadata={'xresolution':'4.25','yresolution':'0.0976','PixelAspectRatio':'43.57'}
        # tifffile.imsave(path, img, imagej=imagej, metadata={})
        raise NotImplementedError("Saving of metadata is not yet implemented")

def concatenate_z(stack):
    """
    Concatenate in z direction for area mode 'line' or 'kymograph',e.g. coronal section. 
    This is necessary because z steps are otherwise treated as additional temporal frame, 
    i.e. in Fiji the frames jump up and down between z positions.

    Parameters
    ----------
    stack : 4D or 6D numpy array
        Stack to be z concatenated.

    Returns
    -------
    stack : 3D or 5D numpy array
        Concatenated stack.

    Examples
    --------
    >>> import utils2p
    >>> import numpy as np
    >>> stack = np.zeros((100, 2, 64, 128))
    >>> concatenated = utils2p.concatenate_z(stack)
    >>> concatenated.shape
    (100, 128, 128)
    """
    res = np.concatenate(np.split(stack, stack.shape[-3], axis=-3), axis=-2)
    return np.squeeze(res)



# FUNCTIONS TO PROCESS DATA

def create_tiffs(directory):
    """
    Given a folder containing .raw data and .xml metadata,
    load the raw data, then save it as a single tiff stack.
    If two channels are given, they are saved as separate tiffs.

    Parameters
    ----------
    directory : str
        Path to directory containing files
    """
    # find files
    raw_path = find_raw_file(directory)
    raw_directory = os.path.dirname(raw_path)
    metadata_path = find_metadata_file(directory)
    metadata_directory = os.path.dirname(metadata_path)
    # check that files are in the same directory
    assert raw_directory == metadata_directory, (
        'Found .raw and .xml files in different directories:'
        ' {} vs {}'.format(raw_directory, metadata_directory)
    )
    # get imaging metadata
    metadata = Metadata(metadata_path)

    # load raw data
    stacks = load_raw(raw_path, metadata)

    # convert to tiff (1 or 2 stacks)
    if len(stacks) == 1:
        save_img(os.path.join(raw_directory, 'channel_1.tif'), stacks[0])
    elif len(stacks) == 2:
        save_img(os.path.join(raw_directory, 'channel_1.tif'), stacks[0])
        save_img(os.path.join(raw_directory, 'channel_2.tif'), stacks[1])
    else:
        raise ValueError('Expected 1 or 2 stacks, got {}'.format(len(stacks)))


