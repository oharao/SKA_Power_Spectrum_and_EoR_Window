#!/usr/bin/env python
import sys

import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy
from scipy.fft import fftfreq
import csv
from scipy.signal import windows
from scipy import stats

def radial_profile(image, centre=None):
    if centre is None:
        centre = (image.shape[0] // 2, image.shape[1] // 2)
    x, y = numpy.indices((image.shape))
    r = numpy.sqrt((x - centre[0])**2 + (y - centre[1])**2)
    r = r.astype(numpy.int)
    return numpy.bincount(r.ravel(), image.ravel()) / numpy.bincount(r.ravel())


def get_k_parallel(z, tau_vec):
    omega_m = 0.3156
    omega_lambda = 0.6844
    omega_r = 8e-5
    E_z = np.sqrt(omega_r * (1 + z) ** 4 + omega_m * (1 + z) ** 3 + omega_lambda)
    k_parallel = (np.divide(2 * np.pi * tau_vec * 1420.0e6 * 1e5 * E_z, 3e8 * (1 + z) ** 2))
    return k_parallel


def get_baseline(angular_scale, freq, scale_factor):
    baseline = 1.22 * 3e8 / (freq * angular_scale * 0.0174533)
    baseline_block_boundaries = np.power(np.linspace(np.power(baseline[0], 1 / float(scale_factor)),
                                                     np.power(baseline[-1], 1 / float(scale_factor)),
                                                     num=len(angular_scale)), scale_factor)
    return baseline_block_boundaries


def get_k_perp(baseline_length, freq, Dc):
    k_perp = np.divide(2 * np.pi * freq * baseline_length, 3e8 * Dc)
    return k_perp


def get_Dc_values(Dc_file):
    with open(Dc_file, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            Dc = row
    Dc_values = np.array(Dc, dtype=np.float64)
    return Dc_values


def main(file_name, k=0):
    fits_file = fits.open(file_name)
    # pylint: disable=no-member
    image = fits_file[0].data
    cellsize = abs(fits_file[0].header['CDELT1']) * numpy.pi / 180.0
    spectrum = numpy.abs(numpy.fft.fftshift(numpy.fft.fft2(image)))**2
    profile = radial_profile(spectrum)
    # Delta_u = 1 / (N * Delta_theta)
    cellsize_uv = 1.0 / (2.0 * image.shape[0] * cellsize)
    lambda_max = cellsize_uv * len(profile)
    lambda_axis = numpy.linspace(cellsize_uv, lambda_max, len(profile))
    theta_axis = 180.0 / (numpy.pi * lambda_axis)
    return profile, theta_axis

if __name__ == '__main__':
    freq_values = np.arange(0.130e9, 0.17e9, 0.000012e9)
    freq_interval = 0.000012e9
    delay_values = np.sort(fftfreq(len(freq_values), freq_interval))
    z_values = 1420.0e6 / freq_values - 1

    root = 'Data_for_Oscar/' #'SKA_Power_Spectrum_and_EoR_Window/comoving/130-170MHz/'

    ps = np.zeros([len(freq_values), 182])
    for k in range(len(freq_values)):
        str_freq = format(freq_values[k] * 1e-6, ".3f")
        file_path = root + 'freq_' + str_freq + '_MHz_interpolate_T21_slices.fits'
        ps[k, :] = main(file_path, freq_interval)[0]

    theta_axis = main(file_path)[1]

    k_parallel = get_k_parallel(z_values, delay_values)

    Dc_file = root + '/los_comoving_distance.csv'
    Dc = get_Dc_values(Dc_file)

    baseline_lengths = get_baseline(theta_axis, freq_values[-1], 2)
    k_perp = get_k_perp(baseline_lengths, freq_values[-1], Dc[-1])

    P_d = ps[:, 1:-1].transpose()

    xx, yy = np.meshgrid(k_perp, k_parallel)
    k = np.sqrt(yy ** 2 + xx ** 2)
    delta = ps * k ** 3 / (2 * np.pi ** 2)
    bins = stats.binned_statistic(np.reshape(k.transpose(), -1), np.reshape(delta.transpose(), -1), statistic='mean',
                                  bins=500)
    plt.clf()
    c = plt.loglog(bins.bin_edges[:-1], bins.statistic)
    plt.xlabel('$k$')
    plt.ylabel('$k_\parallel [h Mpc^{-1}]$')
    plt.savefig('1D_ps.png')
    plt.clf()

    plt.clf()
    c = plt.pcolormesh(k_parallel, k_perp[1:-1],  ps[:, 1:-1].transpose(), norm=LogNorm(), cmap='gnuplot')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('$k_\perp [h Mpc^{-1}]$')
    plt.ylabel('$k_\parallel [h Mpc^{-1}]$')
    plt.xlim(0.0071873657302871305, 1.3081005629122575)
    plt.ylim(8e-2, 1.3e0)
    plt.colorbar(c)
    plt.savefig('2D_ps.png')
