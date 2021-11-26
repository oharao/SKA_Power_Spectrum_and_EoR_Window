"""
Newer version
Ran inside casa [Open casa, then execfile('filename.py')]
blocks means binning
reduced refers to attempt at averging abs(v^2) over the +ve/-ve delay
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import fft, fftfreq
import csv
from matplotlib.colors import LogNorm
from bisect import bisect_left
import pandas as pd
# For testing in Pycharm
# import casacore.tables as tb


# take in freq in Hz
def delay_transform(name1, name2, filepath, row, N, freq_values, s_21):
    vis_data = np.zeros([1480, N], dtype=complex)
    vis_data_test = np.zeros([1480, N], dtype=complex)

    gain_data = pd.read_csv('./SKA_Power_Spectrum_and_EoR_Window/s_21/omega=7000_S21_1st.txt', delimiter=' ',
                            names=['freq', 's_21'])
    gain = gain_data['s_21'][0:1480].values

    for k in range(1480):
        freq = '%0.3f' % float((freq_values / 1e6)[k])
        filename = filepath + name1 + freq + name2

        # Comment for Pycharm testing
        tb.open(filename)
        vis = tb.getcol("DATA", row, N)  # structure: [pol,baseline]
        # Uncomment for Pycharm testing
        # table = tb.table(filename)
        # vis = table.getcol("DATA", row, N)[:, 0, :].transpose() # structure: [pol,baseline]

        if s_21 == True:
            vis_data_test[k, :] = vis[0, :]
            vis = vis[0, :] * gain[k] ** 2  # take XX pol for now
        else:
            vis = vis[0, :]
        vis_data[k, :] = vis

    window_vector = np.hanning(1480)  # np.kaiser(40,window_beta)
    window_array = np.tile(window_vector, (N, 1)).T
    window_norm = np.sum(window_vector)
    vis_data2 = vis_data * window_array / window_norm
    vis_delay = fft(vis_data2, axis=0)  # normalisation by 1/N because norm=forward/backwards does not work
    return vis_delay


# used to be the same function as delay_transform but separated to speed it up
def get_baselines_mag(name1, name2, filepath, row, N, freq=80.0):
    freq = '%0.3f' % float(freq)
    filename = filepath + name1 + freq + name2

    # Comment for Pycharm testing
    tb.open(filename)
    uvw_data = tb.getcol("UVW", row, N)  # struture: uvw, baseline
    # Uncomment for Pycharm testing
    # table = tb.table(filename)
    # uvw_data = table.getcol("UVW", row, N).transpose()  # struture: uvw, baseline

    baseline_mag = np.linalg.norm(uvw_data, axis=0)
    return baseline_mag


# abs tau
def get_delay_times(freq, freq_interval):
    delay_time = fftfreq(len(freq), freq_interval)
    """ transferred delay_time = np.abs(delay_time)
    and delay_time = np.delete(delay_time,0,0)
    to P_d functions"""
    # delay_time = fftshift(delay_time) #dont use this if deleting
    return delay_time


# takes in single value z and tau vector,
def get_k_parallel(z, tau_vec):
    omega_m = 0.3156
    omega_lambda = 0.6844
    omega_r = 8e-5
    E_z = np.sqrt(omega_r * (1 + z) ** 4 + omega_m * (1 + z) ** 3 + omega_lambda)
    k_parallel = (np.divide(2 * np.pi * tau_vec * 1420.0e6 * E_z, 3000 * (1 + z) ** 2))
    return k_parallel


def get_Ez(z):
    omega_m = 0.3156
    omega_lambda = 0.6844
    omega_r = 8e-5
    Ez = np.sqrt(omega_r * (1 + z) ** 4 + omega_m * (1 + z) ** 3 + omega_lambda)
    return Ez


def get_k_perp(baseline_mag_vec, freq, Dc):
    k_perp = np.divide(2 * np.pi * freq * baseline_mag_vec, 3e8 * Dc)
    return k_perp


# linear binning in literature
def get_vis_boundaries(sorted_baseline_mag, N=10):
    baseline_block_boundaries = np.linspace(sorted_baseline_mag[0],
                                            sorted_baseline_mag[-1], num=N + 1)
    vis_position = np.zeros(N + 1, dtype=int)
    for i in range(N + 1):
        vis_position[i] = bisect_left(sorted_baseline_mag, baseline_block_boundaries[i])
    return vis_position, baseline_block_boundaries


def get_Dc_values(Dc_file):
    with open(Dc_file, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            Dc = row
    Dc_values = np.array(Dc, dtype=np.float64)
    return Dc_values


def get_delta_Dc_values(delta_Dc_file):
    with open(delta_Dc_file, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            deltaDc = row
    delta_Dc_values = np.array(deltaDc, dtype=np.float64)
    return delta_Dc_values


# same as Pd_avg_unfolded but with binning
def get_Pd_avg_unfolded_binning(name1, name2, filepath, N_baselines, freq_values, freq_interval,
                                time_samples, Dc, delta_Dc, wavelength, z, N_bins, s_21=False):
    # every 130816th baselines are of the same magnitude because it is the same pair rotated around Earth
    baseline_mag = get_baselines_mag(name1, name2, filepath, 0, N_baselines)
    sorted_baseline_mag = np.sort(baseline_mag)
    delay_values = get_delay_times(freq_values, freq_interval)
    sorted_delay_values = np.sort(delay_values)
    vis_position = get_vis_boundaries(sorted_baseline_mag, N_bins)[0]

    baseline_block_boundaries = get_vis_boundaries(sorted_baseline_mag, N_bins)[1]
    sum_sorted_vis = np.zeros([len(sorted_delay_values), N_bins])

    for j in np.arange(time_samples):
        vis_data = delay_transform(name1, name2, filepath, j * N_baselines, N_baselines, freq_values,
                                   s_21=s_21)
        vis_delay = np.abs(vis_data) ** 2  # get modulus squared
        # try sorting by baseline magnitude
        sortedi_vis_delay = vis_delay[:, baseline_mag.argsort()]
        # sort by delay values
        sorted_vis_delay = sortedi_vis_delay[delay_values.argsort(), :]

        sorted_vis_delay_bins = np.zeros([1480, N_bins])
        for q in range(N_bins):
            if vis_position[q] != vis_position[q + 1]:
                sorted_vis_delay_bins[:, q] = np.sum(sorted_vis_delay[:, vis_position[q]:vis_position[q + 1] - 1],
                                                     axis=-1) / (vis_position[q + 1] - vis_position[q])

        sum_sorted_vis = sum_sorted_vis + sorted_vis_delay_bins

    avg_sorted_vis = sum_sorted_vis / time_samples

    k_B = 1.38e-23  # Boltzman constant
    # delayed power spectrum, also A_e?? 40*N of gleam
    P_d = avg_sorted_vis * 1e-52 * 1e3 * np.divide((Dc ** 2) * delta_Dc * wavelength ** 2, (2 * k_B) ** 2) * 1e6
    # eloy said A/T is 1000m^2, and conversion from Jy gives the power of -52
    k_parallel = get_k_parallel(z, sorted_delay_values)
    k_interval = k_parallel[2] - k_parallel[1]
    k_parallel_plot = np.arange(k_parallel[0] - k_interval / 2, k_parallel[-1] + k_interval * 1 / 2,
                                k_interval)  # note I changed the end interval to 1/2 here.#
    k_perp = get_k_perp(baseline_block_boundaries, freq_values[20], Dc)

    return P_d, k_parallel_plot, k_perp


def get_limits(signal, Dc_values, z_values, wavelength_values):
    P_d_gleam, k_parallel_plot, k_perp_plot = signal[0], signal[1], signal[2]

    horizon_limit_gradient = Dc_values[20] * get_Ez(z_values[20]) / (3000 * (1 + z_values[20]))
    horizon_limit_x = np.arange(k_perp_plot.min(), k_perp_plot.max(), 0.1)
    horizon_limit_y = horizon_limit_x * horizon_limit_gradient
    beam_limit_gradient = horizon_limit_gradient * wavelength_values[20] / 38
    beam_limit_y = beam_limit_gradient * horizon_limit_x
    horizon_limit_y_neg = -horizon_limit_x * horizon_limit_gradient
    beam_limit_y_neg = -beam_limit_gradient * horizon_limit_x
    return horizon_limit_x, horizon_limit_y, horizon_limit_y_neg, beam_limit_y, beam_limit_y_neg


def plot_log(limits, gleam, signal, name):
    horizon_limit_x, horizon_limit_y, horizon_limit_y_neg, beam_limit_y, beam_limit_y_neg = limits
    P_d_gleam, k_parallel_plot, k_perp_plot = gleam[0], gleam[1], gleam[2]
    masked_P_d_gleam = np.ma.masked_equal(signal, 0.0, copy=False)

    fig, ax = plt.subplots()
    c = ax.pcolormesh(k_perp_plot[:-1], k_parallel_plot[21:], signal[21:, :],
                      norm=LogNorm(vmin=10 ** 0, vmax=10 ** 14), cmap="jet")
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('$k_\perp [h Mpc^{-1}]$')
    ax.set_ylabel('$k_\parallel [h Mpc^{-1}]$')
    fig.colorbar(c, label='$P_d$ $[mK^2(Mpc/h)^3]$')

    ax.plot(horizon_limit_x, horizon_limit_y, color='white')
    ax.plot(horizon_limit_x, beam_limit_y, color='black')
    ax.plot(horizon_limit_x, horizon_limit_y_neg, color='white')
    ax.plot(horizon_limit_x, beam_limit_y_neg, color='black')
    ax.set_ylim(k_parallel_plot[21:].min(), k_parallel_plot.max())
    ax.set_xlim(k_perp_plot.min(), k_perp_plot.max())
    plt.savefig(name)


def plot_lin(limits, gleam, signal, name):
    horizon_limit_x, horizon_limit_y, horizon_limit_y_neg, beam_limit_y, beam_limit_y_neg = limits
    P_d_gleam, k_parallel_plot, k_perp_plot = gleam[0], gleam[1], gleam[2]
    masked_P_d_gleam = np.ma.masked_equal(signal, 0.0, copy=False)

    fig2, ax2 = plt.subplots()
    c = ax2.pcolormesh(k_perp_plot[:-1], k_parallel_plot, signal,
                       norm=LogNorm(vmin=10 ** 0, vmax=10 ** 14), cmap="jet")
    ax2.set_ylim(k_parallel_plot.min(), k_parallel_plot.max())
    ax2.set_xlim(k_perp_plot.min(), 3.8)
    ax2.set_xlabel('$k_\perp [h Mpc^{-1}]$')
    ax2.set_ylabel('$k_\parallel [h Mpc^{-1}]$')
    fig2.colorbar(c, label='$P_d$ $[mK^2(Mpc/h)^3]$')

    ax2.plot(horizon_limit_x, horizon_limit_y, color='white')
    ax2.plot(horizon_limit_x, beam_limit_y, color='black')
    ax2.plot(horizon_limit_x, horizon_limit_y_neg, color='white')
    ax2.plot(horizon_limit_x, beam_limit_y_neg, color='black')
    plt.savefig(name)


def main():
    freq_values = np.arange(72.0e6, 109.0e6, 0.025e6)  # change this if channels change
    freq_interval = 0.025e6  # change this if channels change

    wavelength_values = 3e8 / freq_values
    z_values = 1420.0e6 / freq_values - 1
    delay_values = get_delay_times(freq_values, freq_interval)

    # filepath1 = "./SKA_Power_Spectrum_and_EoR_Window/SKA_sim_data/test_eor2/"
    filepath1 = "./SKA_Power_Spectrum_and_EoR_Window/SKA_sim_data/21cmBiBox_gleam_ms/"
    filepath2 = "./SKA_Power_Spectrum_and_EoR_Window/SKA_sim_data/MS_EDGES/"
    gleam_name1 = "gleam_all_freq_"
    gleam_name2 = "_MHz.ms"
    signal1 = "CMB_0.4_33.5_5_25000_Freq_"
    signal2 = ".ms"
    control1 = "CMB_0.4_33.5_5_1_Freq_"
    control2 = ".ms"

    Dc_file = './SKA_Power_Spectrum_and_EoR_Window/SKA_sim_data/21cmBiBox/los_comoving_distance.csv'
    Dc_values = get_Dc_values(Dc_file)

    delta_Dc_file = './SKA_Power_Spectrum_and_EoR_Window/SKA_sim_data/21cmBiBox/delta_los_comoving_distance.csv'
    delta_Dc_values = get_delta_Dc_values(delta_Dc_file)

    bandwidth_values = 0.025e6  # bandwidth in Hz

    # now try one channel only, can probably loop over other channels later
    time_samples = 1  # 40  # number of time samples to mod average over
    N_baselines = 130816
    N_bins = 10000
    window_beta = 6

    gleam = get_Pd_avg_unfolded_binning(gleam_name1, gleam_name2, filepath1, N_baselines, freq_values,
                                        freq_interval, time_samples, Dc_values[20], delta_Dc_values[20],
                                        wavelength_values[20],
                                        z_values[20], N_bins, s_21=False)

    gleam_with = get_Pd_avg_unfolded_binning(gleam_name1, gleam_name2, filepath1, N_baselines, freq_values,
                                             freq_interval, time_samples, Dc_values[20], delta_Dc_values[20],
                                             wavelength_values[20],
                                             z_values[20], N_bins, s_21=True)

    """
    Pd_signal = get_Pd_avg_unfolded_binning(signal1, signal2, filepath2, N_baselines, freq_values,
                                            freq_interval, time_samples, Dc_values[20], delta_Dc_values[20],
                                            wavelength_values[20],
                                            z_values[20], N_bins)[0]

    Pd_signal_with = get_Pd_avg_unfolded_binning(signal1, signal2, filepath2, N_baselines, freq_values,
                                            freq_interval, time_samples, Dc_values[20], delta_Dc_values[20],
                                            wavelength_values[20],
                                            z_values[20], N_bins, s_21=True)[0]

    Pd_signalc = get_Pd_avg_unfolded_binning(control1, control2, filepath2, N_baselines, freq_values,
                                             freq_interval, time_samples, Dc_values[20], delta_Dc_values[20],
                                             wavelength_values[20],
                                             z_values[20], N_bins)[0]

    Pd_signalc_with = get_Pd_avg_unfolded_binning(control1, control2, filepath2, N_baselines, freq_values,
                                             freq_interval, time_samples, Dc_values[20], delta_Dc_values[20],
                                             wavelength_values[20],
                                             z_values[20], N_bins, s_21=True)[0]

    Pd_relative = np.divide(Pd_signal, Pd_signalc, out=np.zeros_like(Pd_signal), where=Pd_signalc != 0)
    Pd_relative_with = np.divide(Pd_signal_with, Pd_signalc_with, out=np.zeros_like(Pd_signal_with),
                                 where=Pd_signalc_with != 0)
    """

    limits = get_limits(gleam, Dc_values, z_values, wavelength_values)
    limits_with = get_limits(gleam_with, Dc_values, z_values, wavelength_values)

    plot_log(limits, gleam, gleam[0], "part2_1.png")
    plot_lin(limits, gleam, gleam[0], "part2_2.png")
    plot_log(limits_with, gleam_with, gleam_with[0], "part2_3.1.png")
    plot_lin(limits_with, gleam_with, gleam_with[0], "part2_4.1.png")
    """
    plot_log(limits, gleam, Pd_signal, "part2_3.png")
    plot_lin(limits, gleam, Pd_signal, "part2_4.png")
    plot_log(limits, gleam, Pd_relative, "part2_5.png")
    plot_lin(limits, gleam, Pd_relative, "part2_6.png")
    
    
    plot_log(limits_with, gleam_with, Pd_relative_with, "part2_5.1.png")
    plot_lin(limits_with, gleam_with, Pd_relative_with, "part2_6.1.png")
    """

    diff = np.divide(np.abs(np.subtract(gleam[0], gleam_with[0], out=np.zeros_like(gleam[0]))), gleam[0],
                     out=np.zeros_like(gleam[0]))
    P_d_gleam, k_parallel_plot, k_perp_plot = gleam[0], gleam[1], gleam[2]

    fig, ax = plt.subplots()
    c = ax.pcolormesh(k_perp_plot[:-1], k_parallel_plot, np.log(diff), cmap="jet")
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('$k_\perp [h Mpc^{-1}]$')
    ax.set_ylabel('$k_\parallel [h Mpc^{-1}]$')
    fig.colorbar(c, label='Log($P_d$ $[mK^2(Mpc/h)^3]$)')

    ax.set_ylim(3*10**-3, k_parallel_plot.max())
    ax.set_xlim(k_perp_plot.min(), k_perp_plot.max())

    plt.savefig("residual.png")


if __name__ == '__main__':
    main()
else:
    main()  # doing from casabrowser
