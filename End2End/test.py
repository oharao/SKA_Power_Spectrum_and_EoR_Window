from datetime import datetime
import os

from generate_EoR import plot_eor
import numpy as np

max_freq = 0.265  # GHz (0.1000852)
min_freq = 0.215  # GHz
channel_bandwidth = 0.0001098
channels = 456

observation_num_time_steps = 1

z_l = 55
dc_path = '130-170MHz'
stations = 'antenna_pos/'

date = '20230901_135406360424'
#os.mkdir(date + '_results')


limits, ps_dir, result_dir, delays, baselines = plot_eor(date + '_vis', date + '_results', min_freq, max_freq, channels,
                                                         channel_bandwidth, observation_num_time_steps)


