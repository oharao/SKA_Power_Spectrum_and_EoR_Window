from datetime import datetime
import os

from generate_EoR import plot_eor
import numpy as np

max_freq = 0.165  # GHz (0.1000852)
min_freq = 0.155  # GHz
channel_bandwidth = 0.000125
channels = 80

observation_num_time_steps = 2

z_l = 55
dc_path = '130-170MHz'
stations = 'antenna_pos_core/'

date = 'Paper_Results/Foreground_145-165'
date = 'Paper_Results/EoR_145-165'
date = '20230720_160805258736'
#os.mkdir(date + '_results')


limits, ps_dir, result_dir, delays, baselines = plot_eor(date + '_vis', date + '_results', min_freq, max_freq, channels,
                                                         channel_bandwidth, observation_num_time_steps)


