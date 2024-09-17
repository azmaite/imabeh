""" 
Functions to work with static 2p images, i.e. images that are not part of a time series.
For example, a z stack of a neuron with 3 replicates.

Includes functions:
    flatten_stack() - converts a set of repeat Z-stacks into a single average projection image
"""

import numpy as np
import os
from pystackreg import StackReg

from imabeh.imaging2p import utils2p

# IMPORT ALL PATHS FROM USERPATHS - DO NOT add any paths outside of this import 
from imabeh.run.userpaths import user_config, LOCAL_DIR

def flatten_stack_std(stack_path : str):
    """ 
    Function to flatten a set of repeat Z-stacks into a single average projection image.
    It will:
    - get a STD projection image out of each set of Z-stacks
    - register each projection using StackReg
    - gets an average image of the registered projections
    - saves the final image (add _STD to the file name)

    Parameters
    ----------
    stack_path : str 
        path to a 4D np array (tif): time (reps), z, x, y
    
    Saves as tif
    ----------
    stack_avg : a 2D np array: x, y
    """

    # load the stack tiff file
    if os.path.isfile(stack_path):
        stack = utils2p.load_img(stack_path, memmap=True)
    else:
        raise ValueError('stack_file path is not a valid file')

    # check correct dimensions
    if not len(stack.shape) == 4:
        raise ValueError('stack file must have 4 dimensions')

    # get std projection across z (better than max projection)
    stack_project = np.memmap.std(stack,1)
    # delete stack to save memory
    del(stack)

    # register each projected stack to the first
    sr = StackReg(StackReg.RIGID_BODY)
    stack_reg = sr.register_transform_stack(stack_project, reference='previous')
    del(stack_project)

    # get average across projected stacks
    stack_avg = np.memmap.mean(stack_reg,0)
    del(stack_reg)

    # save stack - add _STD to name
    name, ext = os.path.splitext(stack_path) 
    save_path = name + '_STD' + ext
    utils2p.save_img(save_path, stack_avg)






