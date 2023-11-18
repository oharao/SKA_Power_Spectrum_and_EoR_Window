"""
Create a relevant files for the inclusion of the system noise which are saved to the relevant directories within the
telescope model. Numerical values obtained via SKA paper: https://arxiv.org/ftp/arxiv/papers/1912/1912.12699.pdf

@author:
    Oscar Sage David O'Hara
@email:
    osdo2@cam.ac.uk
"""

import numpy as np
import csv
import astropy.constants as const
import astropy.units as u
from scipy import interpolate

max_freq = 265e6
min_freq = 215e6
bandwidth = 109.8e3

freq_values_Hz = np.arange(min_freq, max_freq, bandwidth)


def read_natural_sensitivities():
    with open(f"SKA_Power_Spectrum_and_EoR_Window/End2End/noise_model/SKA1-Low_Natural_Sensitivities.csv", 'r') as file:
        reader = csv.reader(file)
        data = np.array(list(reader), dtype=float)
    frequencies = data[:,0]
    nat_sens = data[:,1]
    return frequencies, nat_sens


def interp_sensitivities(data, kind='linear'):
    interp_f = interpolate.interp1d(data[:, 0] * 1e6, data[:, 1], kind=kind)
    return interp_f


def get_distribution(natural_sensitivities, bandwidth, system_efficency=0.9, t_acc=0.9):
    rms = const.k_B * np.sqrt(np.divide(2, (natural_sensitivities * u.m ** 2 / u.K
                                            ) ** 2 * system_efficency ** 2 * (bandwidth / u.s) * (t_acc * u.s)))
    return rms / (1e-26 * u.J / (u.m ** 2)) * u.Jansky



def write_freq_rms(rms_name='rms.txt', freq_name='noise_frequencies.txt', bandwidth, integration_time, interp_func=None):

    rms_interp = interp_func

    np.savetxt('rms.txt', get_distribution(rms_interp, bandwidth, t_acc=3.6e6).value)
    np.savetxt('noise_frequencies.txt', freq_values_Hz)
