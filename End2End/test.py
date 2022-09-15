from datetime import datetime
import os

from generate_EoR import plot_eor

max_freq = 0.1601  # GHz (0.1000852)
min_freq = 0.130  # GHz
channel_bandwidth = 0.0001098
channels = 275

date = '20220914_091142'

control = '130-160MHz_control.vis'
#os.mkdir(date + '_results')

plot_eor(control, date + '_vis', date + '_results',
         min_freq, max_freq, channels, channel_bandwidth, '130-160MHz')

