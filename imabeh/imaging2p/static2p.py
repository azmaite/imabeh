""" 
Functions to work with static 2p images, i.e. images that are not part of a time series.
For example, a z stack of a neuron with 3 replicates.
"""

from imabeh.imaging2p import utils2p

# IMPORT ALL PATHS FROM USERPATHS - DO NOT add any paths outside of this import 
from imabeh.run.userpaths import user_config, LOCAL_DIR