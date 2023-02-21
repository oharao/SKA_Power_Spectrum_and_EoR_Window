from datetime import datetime
import os

from generate_EoR import plot_eor
import numpy as np

max_freq = 0.165  # GHz (0.1000852)
min_freq = 0.145  # GHz
channel_bandwidth = 0.000125
channels = 160

z_l = 55
dc_path = '130-170MHz'
stations = 'antenna_pos_core/'

date = 'Paper_Results/Foreground_145-165'
date = 'Paper_Results/EoR_145-165'
#os.mkdir(date + '_results')


limits, ps_dir, result_dir, delays, baselines = plot_eor(date + '_vis', date + '_results', min_freq, max_freq, channels, channel_bandwidth, dc_path)

ps_data = np.load(ps_dir, allow_pickle=True)
eor_data = np.load('pipeline_data/EoR_' + dc_path + '_results/Delta_PS.npy', allow_pickle=True)
foreground_data = np.load('pipeline_data/Foregrounds_' + dc_path + '_results/Delta_PS.npy',
                          allow_pickle=True)
foreground_subtracted_ps = (np.array(ps_data[0] - foreground_data[0]), foreground_data[1], foreground_data[2])
plot_symlognorm(limits, foreground_subtracted_ps, result_dir + 'eor_diff.png', delays, baselines)
np.save(result_dir + 'Foreground_Subtracted_PS.npy', foreground_subtracted_ps)

eor_diff = (np.array(eor_data - foreground_subtracted_ps,), eor_data[1], eor_data[2])
plot_symlognorm(limits, eor_diff, result_dir + 'eor_diff.png', delays, baselines)
np.save(result_dir + 'EoR_diff_PS.npy', eor_diff, dtype=object)
