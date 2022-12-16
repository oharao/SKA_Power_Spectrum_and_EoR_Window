from datetime import datetime
import os

from generate_EoR import plot_eor

max_freq = 0.1001  # GHz (0.1000852)
min_freq = 0.070  # GHz
channel_bandwidth = 0.000012
channels = 2509

date = '20221205_160935009949'
control = '70-100MHz_eor_control.vis'
control = '70-100MHz_control.vis'
#os.mkdir(date + '_results')

plot_eor(control, date + '_vis', date + '_results',
         min_freq, max_freq, channels, channel_bandwidth, 'test')

