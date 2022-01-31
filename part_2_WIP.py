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
import casacore.tables as tb
import pandas as pd


# take in freq in Hz
def delay_transform(name1, name2, filepath, row, N, freq_values, window_beta):
    vis_data = np.zeros([40, N], dtype=complex)
    freq_MHz = freq_values/1e6

    gain_data = pd.read_csv('10m_S21_parameter.txt', delimiter=' ', names=['freq', 's_21'])
    gain = gain_data['s_21'][300:340].values

    for k in range(40):
        freq = '%0.3f' % float(freq_MHz[k])
        filename = filepath + name1 + freq + name2
        table = tb.table(filename)
        vis = table.getcol("DATA", row, N)[:, 0, :].transpose() # structure: [pol,baseline]
        vis = vis[0, :] * gain[k] ** 2 # take XX pol for now
        vis_data[k, :] = vis

    window_vector = np.hanning(40)  # np.kaiser(40,window_beta)
    window_array = np.tile(window_vector, (N, 1)).T
    window_norm = np.sum(window_vector)
    vis_data2 = vis_data * window_array / window_norm
    vis_delay = fft(vis_data2, axis=0)  # normalisation by 1/N because norm=forward/backwards does not work
    """transfer vis_delay = np.delete(vis_delay,0,0) to P_d functions"""
    # vis_delay = fftshift(vis_delay,axes=0) #in delay space, after deleting a
    # row cannot use this anymore
    return vis_delay


# Read in UVW data for calculation of linear norm baseline magnitude
def get_baselines_mag(name1, name2, filepath, row, N, freq=80.0):
    freq = '%0.3f' % float(freq)
    filename = filepath + name1 + freq + name2
    table = tb.table(filename)
    uvw_data = table.getcol("UVW", row, N).transpose()  # struture: uvw, baseline
    baseline_mag = np.linalg.norm(uvw_data, axis=0)
    return baseline_mag


# Calculate delay times, abs tau
def get_delay_times(freq, freq_interval):
    delay_time = fftfreq(len(freq), freq_interval)
    """ transferred delay_time = np.abs(delay_time)
    and delay_time = np.delete(delay_time,0,0)
    to P_d functions"""
    # delay_time = fftshift(delay_time) #dont use this if deleting
    return delay_time


# takes in single value z and tau vector,
def get_k_parallel(z, tau_vec):
    h = 0.6727
    omega_m = 0.3156
    omega_lambda = 0.6844
    omega_r = 8e-5
    E_z = np.sqrt(omega_r * (1 + z) ** 4 + omega_m * (1 + z) ** 3 + omega_lambda)
    k_parallel = (np.divide(2 * np.pi * tau_vec * 1420.0e6 * E_z, 3000 * (1 + z) ** 2))
    return k_parallel


def get_Ez(z):
    h = 0.6727
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
    # log_max = np.log10(np.amax(sorted_baseline_mag))
    # log_min = np.log10(np.amin(sorted_baseline_mag))
    # log_block_boundaries = np.linspace(log_min,log_max,num=N+1)
    # baseline_block_boundaries = 10**(log_block_boundaries)
    baseline_block_boundaries = np.linspace(sorted_baseline_mag[0],
                                            sorted_baseline_mag[-1], num=N + 1)
    vis_position = np.zeros(N + 1, dtype=int)
    for i in range(N + 1):
        vis_position[i] = bisect_left(sorted_baseline_mag, baseline_block_boundaries[i])
    return vis_position, baseline_block_boundaries


# Read comoving distance values from specified file.
def get_Dc_values(Dc_file):
    with open(Dc_file, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            Dc = row
    Dc_values = np.array(Dc, dtype=np.float64)
    return Dc_values


# Read comoving distance deltas from specified file.
def get_delta_Dc_values(delta_Dc_file):
    with open(delta_Dc_file, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            deltaDc = row
    delta_Dc_values = np.array(deltaDc, dtype=np.float64)
    return delta_Dc_values


# same as Pd_avg_unfolded but with binning
def get_Pd_avg_unfolded_binning(name1, name2, filepath, N_baselines, freq_values, freq_interval,
                                time_samples, Dc, delta_Dc, wavelength, bandwidth, z, window_beta, N_bins):
    # every 130816th baselines are of the same magnitude because it is the same pair rotated around Earth
    baseline_mag = get_baselines_mag(name1, name2, filepath, 0, N_baselines)
    sorted_baseline_mag = np.sort(baseline_mag)
    delay_values = get_delay_times(freq_values, freq_interval)
    sorted_delay_values = np.sort(delay_values)
    vis_position = get_vis_boundaries(sorted_baseline_mag, N_bins)[0]

    baseline_block_boundaries = get_vis_boundaries(sorted_baseline_mag, N_bins)[1]
    sum_sorted_vis = np.zeros([len(sorted_delay_values), N_bins])

    for j in np.arange(time_samples):
        vis_data = delay_transform(name1, name2, filepath, j * N_baselines, N_baselines, freq_values, window_beta)
        vis_delay = np.abs(vis_data) ** 2  # get modulus squared
        # try sorting by baseline magnitude
        sortedi_vis_delay = vis_delay[:, baseline_mag.argsort()]
        # sort by delay values
        sorted_vis_delay = sortedi_vis_delay[delay_values.argsort(), :]

        sorted_vis_delay_bins = np.zeros([40, N_bins])
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
    # P_d = np.delete(P_d,0,0)
    ########## the k axes
    k_parallel = get_k_parallel(z, sorted_delay_values)
    k_interval = k_parallel[2] - k_parallel[1]
    k_parallel_plot = np.arange(k_parallel[0] - k_interval / 2, k_parallel[-1] + k_interval * 1 / 2,
                                k_interval)  # note I changed the end interval to 1/2 here.
    # k_parallel_plot = np.insert(k_parallel_plot,0,0) #bin edge values for pcolormesh
    k_perp = get_k_perp(baseline_block_boundaries, freq_values[20], Dc)
    ##################

    return P_d, k_parallel_plot, k_perp


def main():
    freq_values = np.arange(79.5e6, 80.5e6, 0.025e6)  # change this if channels change
    freq_interval = 0.025e6  # change this if channels change

    wavelength_values = 3e8 / freq_values
    z_values = 1420.0e6 / freq_values - 1

    filepath1 = "./SKA_sim_data/test_eor2/"
    filepath2 = "./SKA_sim_data/MS_EDGES/"
    gleam_name1 = "gleam_all_freq"
    gleam_name2 = "MHz.ms"
    signal1 = "CMB_0.4_33.5_5_25000_Freq_"
    signal2 = ".ms"
    control1 = "CMB_0.4_33.5_5_1_Freq_"
    control2 = ".ms"

    Dc_file = './SKA_sim_data/21cmBiBox/los_comoving_distance.csv'
    Dc_values = get_Dc_values(Dc_file)

    delta_Dc_file = './SKA_sim_data/21cmBiBox/delta_los_comoving_distance.csv'
    delta_Dc_values = get_delta_Dc_values(delta_Dc_file)

    bandwidth_values = 0.025e6  # bandwidth in Hz

    # now try one channel only, can probably loop over other channels later
    time_samples = 40  # number of time samples to mod average over
    N_baselines = 130816
    N_bins = 10000
    window_beta = 6

    gleam = get_Pd_avg_unfolded_binning(gleam_name1, gleam_name2, filepath1, N_baselines, freq_values,
                                        freq_interval, time_samples, Dc_values[20], delta_Dc_values[20],
                                        wavelength_values[20],
                                        bandwidth_values, z_values[20], window_beta, N_bins)

    P_d_gleam = gleam[0]
    k_parallel_plot = gleam[1]
    k_perp_plot = gleam[2]

    Pd_signal = get_Pd_avg_unfolded_binning(signal1, signal2, filepath2, N_baselines, freq_values,
                                            freq_interval, time_samples, Dc_values[20], delta_Dc_values[20],
                                            wavelength_values[20],
                                            bandwidth_values, z_values[20], window_beta, N_bins)[0]

    Pd_signalc = get_Pd_avg_unfolded_binning(control1, control2, filepath2, N_baselines, freq_values,
                                             freq_interval, time_samples, Dc_values[20], delta_Dc_values[20],
                                             wavelength_values[20],
                                             bandwidth_values, z_values[20], window_beta, N_bins)[0]

    Pd_relative = np.divide(Pd_signal, Pd_signalc, out=np.zeros_like(Pd_signal), where=Pd_signalc != 0)

    horizon_limit_gradient = Dc_values[20] * get_Ez(z_values[20]) / (3000 * (1 + z_values[20]))
    horizon_limit_x = np.arange(k_perp_plot.min(), k_perp_plot.max(), 0.1)
    horizon_limit_y = horizon_limit_x * horizon_limit_gradient
    beam_limit_gradient = horizon_limit_gradient * wavelength_values[20] / 38
    beam_limit_y = beam_limit_gradient * horizon_limit_x
    horizon_limit_y_neg = -horizon_limit_x * horizon_limit_gradient
    beam_limit_y_neg = -beam_limit_gradient * horizon_limit_x

    # change all k_para to [21:] if you want to plot log, and P_d_gleam to [21:,:]
    fig, ax = plt.subplots()
    masked_P_d_gleam = np.ma.masked_equal(P_d_gleam, 0.0, copy=False)
    c = ax.pcolormesh(k_perp_plot, k_parallel_plot[21:], P_d_gleam[21:, :],
                      norm=LogNorm(vmin=masked_P_d_gleam.min(), vmax=P_d_gleam.max()), cmap="jet")
    ax.set_ylim(k_parallel_plot[21:].min(), k_parallel_plot.max())
    ax.set_xlim(k_perp_plot.min(), k_perp_plot.max())
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('$k_\perp [h Mpc^{-1}]$')
    ax.set_ylabel('$k_\parallel [h Mpc^{-1}]$')
    fig.colorbar(c, label='$P_d$ $[mK^2(Mpc/h)^3]$')

    ax.plot(horizon_limit_x, horizon_limit_y, color='white')
    ax.plot(horizon_limit_x, beam_limit_y, color='black')
    ax.plot(horizon_limit_x, horizon_limit_y_neg, color='white')
    ax.plot(horizon_limit_x, beam_limit_y_neg, color='black')
    plt.savefig("part2_1.png")

    fig2, ax2 = plt.subplots()
    c = ax2.pcolormesh(k_perp_plot, k_parallel_plot, P_d_gleam,
                       norm=LogNorm(vmin=masked_P_d_gleam.min(), vmax=P_d_gleam.max()), cmap="jet")
    ax2.set_ylim(k_parallel_plot.min(), k_parallel_plot.max())
    ax2.set_xlim(k_perp_plot.min(), 3.7)
    ax2.set_xlabel('$k_\perp [h Mpc^{-1}]$')
    ax2.set_ylabel('$k_\parallel [h Mpc^{-1}]$')
    fig2.colorbar(c, label='$P_d$ $[mK^2(Mpc/h)^3]$')

    ax2.plot(horizon_limit_x, horizon_limit_y, color='white')
    ax2.plot(horizon_limit_x, beam_limit_y, color='black')
    ax2.plot(horizon_limit_x, horizon_limit_y_neg, color='white')
    ax2.plot(horizon_limit_x, beam_limit_y_neg, color='black')
    plt.savefig("part2_2.png")

    fig3, ax3 = plt.subplots()
    masked_Pd_signal = np.ma.masked_equal(Pd_signal, 0.0, copy=False)
    c = ax3.pcolormesh(k_perp_plot, k_parallel_plot[21:], Pd_signal[21:, ],
                       norm=LogNorm(vmin=masked_Pd_signal.min(), vmax=Pd_signal.max()), cmap="jet")
    ax3.set_ylim(k_parallel_plot[21:].min(), k_parallel_plot.max())
    ax3.set_xlim(0.023, 3.8)
    ax3.set_xlabel('$k_\perp [h Mpc^{-1}]$')
    ax3.set_ylabel('$k_\parallel [h Mpc^{-1}]$')
    ax3.set_xscale('log')
    ax3.set_yscale('log')
    fig3.colorbar(c, label='$P_d$ $[mK^2(Mpc/h)^3]$')

    ax3.plot(horizon_limit_x, horizon_limit_y, color='white')
    ax3.plot(horizon_limit_x, beam_limit_y, color='black')
    ax3.plot(horizon_limit_x, horizon_limit_y_neg, color='white')
    ax3.plot(horizon_limit_x, beam_limit_y_neg, color='black')
    plt.savefig("part2_3.png")

    fig4, ax4 = plt.subplots()
    masked_Pd_signal = np.ma.masked_equal(Pd_signal, 0.0, copy=False)
    c = ax4.pcolormesh(k_perp_plot, k_parallel_plot, Pd_signal,
                       norm=LogNorm(vmin=masked_Pd_signal.min(), vmax=Pd_signal.max()), cmap="jet")
    ax4.set_ylim(k_parallel_plot.min(), k_parallel_plot.max())
    ax4.set_xlim(k_perp_plot.min(), 3.8)
    ax4.set_xlabel('$k_\perp [h Mpc^{-1}]$')
    ax4.set_ylabel('$k_\parallel [h Mpc^{-1}]$')
    fig4.colorbar(c, label='$P_d$ $[mK^2(Mpc/h)^3]$')

    ax4.plot(horizon_limit_x, horizon_limit_y, color='white')
    ax4.plot(horizon_limit_x, beam_limit_y, color='black')
    ax4.plot(horizon_limit_x, horizon_limit_y_neg, color='white')
    ax4.plot(horizon_limit_x, beam_limit_y_neg, color='black')
    plt.savefig("part2_4.png")

    fig5, ax5 = plt.subplots()
    masked_Pd_relative = np.ma.masked_equal(Pd_relative, 0.0, copy=False)
    c = ax5.pcolormesh(k_perp_plot, k_parallel_plot[21:], Pd_relative[21:, :],
                       norm=LogNorm(vmin=masked_Pd_relative.min(), vmax=Pd_relative.max()), cmap="jet")
    ax5.set_ylim(k_parallel_plot[21:].min(), k_parallel_plot.max())
    ax5.set_xlim(0.023, 3.8)
    ax5.set_xlabel('$k_\perp [h Mpc^{-1}]$')
    ax5.set_ylabel('$k_\parallel [h Mpc^{-1}]$')
    ax5.set_xscale('log')
    ax5.set_yscale('log')
    fig5.colorbar(c, label='Relative Power')

    ax5.plot(horizon_limit_x, horizon_limit_y, color='white')
    ax5.plot(horizon_limit_x, beam_limit_y, color='black')
    ax5.plot(horizon_limit_x, horizon_limit_y_neg, color='white')
    ax5.plot(horizon_limit_x, beam_limit_y_neg, color='black')
    plt.savefig("part2_5.png")

    fig6, ax6 = plt.subplots()
    masked_Pd_relative = np.ma.masked_equal(Pd_relative, 0.0, copy=False)
    c = ax6.pcolormesh(k_perp_plot, k_parallel_plot, Pd_relative,
                       norm=LogNorm(vmin=masked_Pd_relative.min(), vmax=Pd_relative.max()), cmap="jet")
    ax6.set_ylim(k_parallel_plot.min(), k_parallel_plot.max())
    ax6.set_xlim(k_perp_plot.min(), 3.7)
    ax6.set_xlabel('$k_\perp [h Mpc^{-1}]$')
    ax6.set_ylabel('$k_\parallel [h Mpc^{-1}]$')
    fig6.colorbar(c, label='Relative Power')

    ax6.plot(horizon_limit_x, horizon_limit_y, color='white')
    ax6.plot(horizon_limit_x, beam_limit_y, color='black')
    ax6.plot(horizon_limit_x, horizon_limit_y_neg, color='white')
    ax6.plot(horizon_limit_x, beam_limit_y_neg, color='black')
    plt.savefig("part2_6.png")


if __name__ == '__main__':
    main()
else:
    main()  # doing from casabrowser
