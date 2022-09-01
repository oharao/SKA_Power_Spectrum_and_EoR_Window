from datetime import datetime
import os

from generate_EoR import plot_eor


max_freq = 0.1001  # GHz (0.1000852)
min_freq = 0.070  # GHz
channel_bandwidth = 0.0001098
channels = 275

date = '20220902_150919'

control = 'control.vis'
os.mkdir(date + '_results')

plot_eor(control, date + '_vis', date + '_results',
         min_freq, max_freq, channels, channel_bandwidth)

