"""
blocks means binning
reduced refers to attempt at averging abs(v^2) over the +ve/-ve delay
"""

import csv
import matplotlib.pyplot as plt
import numpy as np
import os
import oskar
from bisect import bisect_left
from matplotlib.colors import LogNorm
from scipy.fft import fft, ifft, fftfreq
from scipy.signal import windows


# take in freq in Hz
def delay_transform(name1, name2, filepath, row, N, freq_values, channels, baseline_mag, control=False):
    """
    Performs a delay transform on visibility data at a given frequency.

    Parameters:
        name1 (str): A string representing the first part of the file name before the frequency value.
        name2 (str): A string representing the second part of the file name after the frequency value.
        filepath (str): A string representing the path to the directory containing the visibility data files.
        row (int): An integer representing the row index of the cross-correlation matrix to use.
        N (int): An integer representing the number of time samples in the visibility data.
        freq_values (np.ndarray): An array of floats representing the frequency values in Hz.
        channels (int): An integer representing the number of frequency channels in the visibility data.
        baseline_mag (np.ndarray): An array of floats representing the magnitude of the baseline used in the cross-correlation.
        control (bool): A boolean value indicating whether to include or exclude baselines with zero magnitude from the output.

    Returns:
        np.ndarray: An array of complex numbers representing the delay transform of the visibility data.

    Notes:
        - The function reads the visibility data files and selects the cross-correlation matrix at the specified frequency and row index.
        - The function sorts the selected visibility data by baseline magnitude.
        - The function applies a Blackman-Harris window to the sorted visibility data and computes the delay transform using the FFT.
        - If control is True, the function returns only the delay transform of the visibility data.
        - If control is False, the function also returns the indices of the first and last non-zero baselines in the sorted visibility data.
    """
    vis_data = np.zeros([channels, N], dtype=complex)

    for k in range(channels):
        freq = '%0.3f' % float((freq_values / 1e6)[k])

        (header, handle) = oskar.VisHeader.read(filepath + "/" + name1 + freq + name2)
        block = oskar.VisBlock.create_from_header(header)
        for i in range(header.num_blocks):
            block.read(header, handle, i)
            vis = block.cross_correlations()
            vis_data[k, :] = (vis[row, 0])[:, 0][baseline_mag.argsort()]

    if control is False:
        baselines_index = np.where(np.abs(vis_data[0]) != 0)
        vis_data = vis_data[:, baselines_index[0][0]:baselines_index[0][-1]]

    window_vector = windows.blackmanharris(channels)
    window_array = np.tile(window_vector ** 2, (len(vis_data[0]), 1)).T
    window_norm = np.sum(window_vector ** 2)
    vis_data2 = vis_data * window_array / window_norm
    vis_delay = fft(vis_data2, axis=0)  # normalisation by 1/N because norm=forward/backwards does not work
    if control is False:
        return vis_delay, baselines_index[0][0], baselines_index[0][-1]
    else:
        return vis_delay


# used to be the same function as delay_transform but separated to speed it up
def get_baselines_mag(name1, name2, filepath, freq_values):
    """
    Calculates the magnitude of baselines for a given frequency channel from simulated visibility data.

    Parameters:
    name1 (str): The name of the simulated visibility data file prefix.
    name2 (str): The name of the simulated visibility data file suffix.
    filepath (str): The file path of the directory where the simulated visibility data is stored.
    freq_values (float): The frequency channel value in Hz.

    Returns:
    numpy.ndarray: A 1D numpy array of baseline magnitudes in meters.

    This function reads simulated visibility data for the given frequency channel, extracts the baseline
    coordinates and calculates the magnitude of each baseline. It then returns an array of baseline magnitudes.
    """
    freq = '%0.3f' % float((freq_values / 1e6)[0])

    (header, handle) = oskar.VisHeader.read(filepath + "/" + name1 + freq + name2)
    block = oskar.VisBlock.create_from_header(header)
    for i in range(header.num_blocks):
        block.read(header, handle, i)
        u = block.baseline_uu_metres()
        v = block.baseline_vv_metres()
        w = block.baseline_ww_metres()
    uvw_data = np.array((u, v, w))  # struture: uvw, baselinene
    return np.linalg.norm(uvw_data, axis=0)


def get_delay_times(freq, freq_interval):
    """
    Calculates the delay times for a given frequency array and frequency interval using the Fast Fourier Transform.

    Parameters:
        freq (numpy.ndarray): The frequency array in Hz.
        freq_interval (float): The frequency interval in Hz.

    Returns:
        numpy.ndarray: An array of delay times in seconds.
    """
    delay_time = fftfreq(len(freq), freq_interval)
    return delay_time


def get_k_parallel(z, tau_vec):
    """
    Calculate the parallel wave vector for a given redshift and tau vector.

    Parameters
    ----------
    z : float
        Redshift value.
    tau_vec : numpy array
        Array of values representing the time delay of a signal.

    Returns
    -------
    k_parallel : numpy array
        Array of values representing the parallel wave vector.

    """
    # Set cosmological parameters
    omega_m = 0.3156
    omega_lambda = 0.6844
    omega_r = 8e-5

    # Calculate E_z
    E_z = np.sqrt(omega_r * (1 + z) ** 4 + omega_m * (1 + z) ** 3 + omega_lambda)

    # Calculate k_parallel
    k_parallel = (np.divide(2 * np.pi * tau_vec * 1420.0e6 * 1e5 * E_z, 3e8 * (1 + z) ** 2))

    return k_parallel


def get_Ez(z):
    """
    Calculate the value of the E(z) function for a given redshift.

    Parameters
    ----------
    z : float
        Redshift value.

    Returns
    -------
    Ez : float
        Value of the E(z) function.

    """
    # Set cosmological parameters
    omega_m = 0.3156
    omega_lambda = 0.6844
    omega_r = 8e-5

    # Calculate the value of E(z)
    Ez = np.sqrt(omega_r * (1 + z) ** 4 + omega_m * (1 + z) ** 3 + omega_lambda)

    return Ez


def get_k_perp(baseline_mag_vec, freq, Dc):
    """
    Calculate the perpendicular wave vector for a given baseline magnitude, frequency, and comoving distance.

    Parameters
    ----------
    baseline_mag_vec : numpy array
        Array of values representing the magnitude of the baseline vector.
    freq : float
        Frequency value.
    Dc : float
        Comoving distance value.

    Returns
    -------
    k_perp : numpy array
        Array of values representing the perpendicular wave vector.

    """
    # Calculate k_perp using the given inputs
    k_perp = np.divide(2 * np.pi * freq * baseline_mag_vec, 3e8 * Dc)

    return k_perp


def get_vis_boundaries(sorted_baseline_mag, N=10):
    """
    Calculate the boundaries for log binning of visibility data.

    Parameters
    ----------
    sorted_baseline_mag : numpy array
        Array of values representing the magnitude of the baseline vector, sorted in ascending order.
    N : int, optional
        The number of visibility bins to use. Default is 10.

    Returns
    -------
    vis_position : numpy array
        Array of integer values representing the indices of the visibility bins in the sorted baseline magnitude array.
    baseline_block_boundaries : numpy array
        Array of values representing the boundaries between visibility bins.

    """
    # Set scale factor for binning
    scale_factor = 2

    # Calculate block boundaries using log binning
    baseline_block_boundaries = np.power(np.linspace(np.power(sorted_baseline_mag[0], 1 / float(scale_factor)),
                                                     np.power(sorted_baseline_mag[-1], 1 / float(scale_factor)),
                                                     num=N + 1), scale_factor)

    # Find the indices of the block boundaries in the sorted baseline magnitude array
    vis_position = np.zeros(N + 1, dtype=int)
    for i in range(N + 1):
        vis_position[i] = bisect_left(sorted_baseline_mag, baseline_block_boundaries[i])

    return vis_position, baseline_block_boundaries


def get_Dc_values(Dc_file):
    """
    Read distance values from a CSV file and return them as a numpy array.

    Parameters
    ----------
    Dc_file : str
        The file name (including the path) for the CSV file containing distance values.

    Returns
    -------
    Dc_values : numpy array
        A numpy array of distance values in float format.

    """
    # Open the CSV file and read the distance values into a list
    with open(Dc_file, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            Dc = row

    # Convert the list of strings to a numpy array of float values
    Dc_values = np.array(Dc, dtype=np.float64)

    return Dc_values


def get_delta_Dc_values(delta_Dc_file):
    """
    Read change in distance values from a CSV file and return them as a numpy array.

    Parameters
    ----------
    delta_Dc_file : str
        The file name (including the path) for the CSV file containing the change in distance values.

    Returns
    -------
    delta_Dc_values : numpy array
        A numpy array of the change in distance values in float format.

    """
    # Open the CSV file and read the change in distance values into a list
    with open(delta_Dc_file, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            deltaDc = row

    # Convert the list of strings to a numpy array of float values
    delta_Dc_values = np.array(deltaDc, dtype=np.float64)

    return delta_Dc_values


def get_Pd_avg_unfolded_binning(name1: str, name2: str, control_path: str, filepath: str, N_baselines: int,
                                freq_values: np.ndarray, freq_interval: float, channels: int, time_samples: int,
                                Dc: np.ndarray, delta_Dc: np.ndarray, wavelength: np.ndarray, z: float,
                                N_bins: int) -> np.ndarray:
    """
    Computes the average power delay spectrum of a radio interferometric observation after binning the baselines.

    Parameters
    ----------
    name1 : str
        File name prefix.
    name2 : str
        File name sufix.
    control_path : str
        Path to the control observation data.
    filepath : str
        Path to the observation data.
    N_baselines : int
        Number of baselines.
    freq_values : np.ndarray
        Array of frequency values.
    freq_interval : float
        Interval between frequency values.
    channels : int
        Number of channels.
    time_samples : int
        Number of time samples.
    Dc : np.ndarray
        Array of distances to the sources.
    delta_Dc : np.ndarray
        Array of distance errors.
    wavelength : np.ndarray
        Array of wavelengths.
    z : float
        Redshift value.
    N_bins : int
        Number of bins to divide the baselines into.

    Returns
    -------
    np.ndarray: Array of the average power delay spectrum after binning.
    """

    # every 130816th baselines are of the same magnitude because it is the same pair rotated around Earth
    baseline_mag = get_baselines_mag(name1, name2, filepath, freq_values)[0]
    sorted_baseline_mag = np.sort(baseline_mag)

    delay_values = get_delay_times(freq_values, freq_interval)
    sorted_delay_values = np.sort(delay_values)

    # divide the baselines into bins based on their magnitude
    vis_position = get_vis_boundaries(sorted_baseline_mag, N_bins)[0]

    sum_sorted_vis = np.zeros([len(sorted_delay_values), N_bins])
    sum_sorted_vis_control = np.zeros([len(sorted_delay_values), N_bins])
    sum_sorted_vis_residual = np.zeros([len(sorted_delay_values), N_bins])

    # compute the power delay spectrum for each time sample
    for j in np.arange(time_samples):
        delay_data, N1, N2 = delay_transform(name1, name2, filepath, j * N_baselines, N_baselines, freq_values,
                                             channels, baseline_mag)
        vis_data = np.zeros((delay_values.shape[0], baseline_mag.shape[0]), dtype='complex_')
        vis_data[:, N1:N2] = delay_data

        delay_data_control, N3, N4 = delay_transform(name1, name2, control_path, j * N_baselines,
                                                     N_baselines, freq_values,
                                                     channels, baseline_mag)
        vis_data_control = np.zeros((delay_values.shape[0], baseline_mag.shape[0]), dtype='complex_')
        vis_data_control[:, N3:N4] = delay_data_control

        N_start = max(N1, N3)
        N_stop = min(N2, N4)

        vis_data_residual = vis_data[:, N_start:N_stop] - vis_data_control[:, N_start:N_stop]

        vis_delay = np.abs(vis_data) ** 2  # get modulus squared
        vis_delay_control = np.abs(vis_data_control[:, N_start:N_stop]) ** 2
        vis_delay_residual = np.abs(vis_data_residual) ** 2

        # sort by delay values
        sorted_vis_delay = vis_delay[delay_values.argsort(), :]
        sorted_vis_delay_control = vis_delay_control[delay_values.argsort(), :]
        sorted_vis_delay_residual = vis_delay_residual[delay_values.argsort(), :]

        sorted_vis_delay_bins = np.zeros([channels, N_bins])
        sorted_vis_delay_bins_control = np.zeros([channels, N_bins])
        sorted_vis_delay_bins_residual = np.zeros([channels, N_bins])
        for q in range(N_bins):
            if vis_position[q] != vis_position[q + 1]:
                sorted_vis_delay_bins[:, q] = np.sum(sorted_vis_delay[:, vis_position[q]:vis_position[q + 1] - 1],
                                                     axis=-1) / (vis_position[q + 1] - vis_position[q])
                sorted_vis_delay_bins_control[:, q] = np.sum(
                    sorted_vis_delay_control[:, vis_position[q]:vis_position[q + 1] - 1], axis=-1) / (
                                                              vis_position[q + 1] - vis_position[q])
                sorted_vis_delay_bins_residual[:, q] = np.sum(
                    sorted_vis_delay_residual[:, vis_position[q]:vis_position[q + 1] - 1], axis=-1) / (
                                                               vis_position[q + 1] - vis_position[q])
        sum_sorted_vis = sum_sorted_vis + sorted_vis_delay_bins
        sum_sorted_vis_control = sum_sorted_vis_control + sorted_vis_delay_bins_control
        sum_sorted_vis_residual = sum_sorted_vis_residual + sorted_vis_delay_bins_residual

    avg_sorted_vis = sum_sorted_vis / time_samples
    avg_sorted_vis_control = sum_sorted_vis_control / time_samples
    avg_sorted_vis_residual = sum_sorted_vis_residual / time_samples

    k_B = 1.38e-23  # Boltzman constant

    # delayed power spectrum, also A_e?? 40*N of gleam
    P_d = (avg_sorted_vis * 1e-52 * 1e6) * (1e6 / (freq_interval ** 2 * wavelength[:, None] ** 2)) * Dc[:,
                                                                                                     None] ** 2 * np.insert(
        np.insert(delta_Dc, -1, delta_Dc[-1]), 0, delta_Dc[0])[:, None] * (
                  wavelength[:, None] ** 2 / (2 * k_B)) ** 2
    P_d_control = (avg_sorted_vis_control * 1e-52 * 1e6) * (1e6 / (freq_interval ** 2 * wavelength[:, None] ** 2)) * Dc[
                                                                                                                     :,
                                                                                                                     None] ** 2 * np.insert(
        np.insert(delta_Dc, -1, delta_Dc[-1]), 0, delta_Dc[0])[:, None] * (
                          wavelength[:, None] ** 2 / (2 * k_B)) ** 2
    P_d_residual = (avg_sorted_vis_residual * 1e-52 * 1e6) * (
                1e6 / (freq_interval ** 2 * wavelength[:, None] ** 2)) * Dc[:, None] ** 2 * np.insert(
        np.insert(delta_Dc, -1, delta_Dc[-1]), 0, delta_Dc[0])[:, None] * (
                           wavelength[:, None] ** 2 / (2 * k_B)) ** 2

    baseline_block_boundaries = get_vis_boundaries(sorted_baseline_mag[N_start:N_stop], N_bins)[1]

    # eloy said A/T is 1000m^2, and conversion from Jy gives the power of -52
    k_parallel = get_k_parallel(z[delay_values.argsort()], sorted_delay_values)

    k_perp = get_k_perp(baseline_block_boundaries, freq_values[int(Dc.shape[0] / 2)], Dc[int(Dc.shape[0] / 2)])

    eor = P_d, k_parallel, k_perp
    eor_control = P_d_control, k_parallel, k_perp
    eor_residual = P_d_residual, k_parallel, k_perp

    return eor, eor_control, eor_residual, sorted_delay_values


def get_limits(signal, Dc_values, z_values, wavelength_values):
    """
    Calculates the horizon and beam limits of a radio telescope observation using GLEAM data.

    Args:
    - signal: tuple of three 1D arrays (P_d_gleam, k_parallel_plot, k_perp_plot) representing
              the GLEAM signal to be analyzed
    - Dc_values: 1D array of comoving distance values in Mpc/h
    - z_values: 1D array of redshift values corresponding to the Dc_values
    - wavelength_values: 1D array of wavelength values in meters corresponding to the z_values

    Returns:
    - horizon_limit_x: 1D array of k_perp_plot values
    - horizon_limit_y: 1D array of horizon limit values for each k_perp_plot value
    - horizon_limit_y_neg: 1D array of negative horizon limit values for each k_perp_plot value
    - beam_limit_y: 1D array of beam limit values for each k_perp_plot value
    - beam_limit_y_neg: 1D array of negative beam limit values for each k_perp_plot value
    """
    # extract GLEAM signal components
    P_d_gleam, k_parallel_plot, k_perp_plot = signal[0], signal[1], signal[2]

    # calculate horizon limit gradient and x values
    horizon_limit_gradient = Dc_values[20] * get_Ez(z_values[20]) / (3000 * (1 + z_values[20]))
    horizon_limit_x = np.arange(k_perp_plot.min(), k_perp_plot.max(), 0.1)

    # calculate horizon limit and beam limit y values
    horizon_limit_y = horizon_limit_x * horizon_limit_gradient
    beam_limit_gradient = horizon_limit_gradient * wavelength_values[20] / 38
    beam_limit_y = beam_limit_gradient * horizon_limit_x
    horizon_limit_y_neg = -horizon_limit_x * horizon_limit_gradient
    beam_limit_y_neg = -beam_limit_gradient * horizon_limit_x

    # return limit values
    return horizon_limit_x, horizon_limit_y, horizon_limit_y_neg, beam_limit_y, beam_limit_y_neg


def plot_log(limits, gleam, signal, name, delay, vmax=1e9, vmin=1e-6, cmap='gnuplot'):
    """
    Plot a logarithmic scale color map of a signal in k-space, along with white and black contours that represent the horizon and beam limits, respectively.

    Parameters:
    -----------
    limits: tuple
        A tuple of the form (horizon_limit_x, horizon_limit_y, horizon_limit_y_neg, beam_limit_y, beam_limit_y_neg)
        where each element is a list of float values representing the limits of the horizon and the beam.
    gleam: tuple
        A tuple of the form (P_d_gleam, k_parallel_plot, k_perp_plot) where P_d_gleam is a 2D array containing
        the power spectrum values, and k_parallel_plot and k_perp_plot are arrays of the k values for the
        power spectrum in the parallel and perpendicular directions, respectively.
    signal: numpy array
        A 2D numpy array containing the signal values.
    name: str
        The name of the output file to be saved.
    delay: numpy array
        A 1D numpy array of delay values.
    vmax: float, optional
        The maximum value of the signal to be displayed, defaults to 1e9.
    vmin: float, optional
        The minimum value of the signal to be displayed, defaults to 1e-6.
    cmap: str, optional
        The color map to be used, defaults to 'gnuplot'.

    """
    horizon_limit_x, horizon_limit_y, horizon_limit_y_neg, beam_limit_y, beam_limit_y_neg = limits
    P_d_gleam, k_parallel_plot, k_perp_plot = gleam[0], gleam[1], gleam[2]
    masked_P_d_gleam = np.ma.masked_equal(signal, 0.0, copy=False)

    fig, ax1 = plt.subplots()
    c = ax1.pcolormesh(k_perp_plot[:-1], k_parallel_plot[:], signal[:, :],
                       norm=LogNorm(), cmap=cmap)
    ax2 = ax1.twinx()

    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax2.set_yscale('log')
    ax1.set_xlabel('$k_\perp [h Mpc^{-1}]$')
    ax1.set_ylabel('$k_\parallel [h Mpc^{-1}]$')
    ax2.set_ylabel('Log(Delay) $[ns]$')

    fig.colorbar(c, label='$P_d$ $[mK^2(Mpc/h)^3]$', pad=0.13)

    ax1.plot(horizon_limit_x, horizon_limit_y, color='white')
    ax1.plot(horizon_limit_x, beam_limit_y, color='black')
    ax1.plot(horizon_limit_x, horizon_limit_y_neg, color='white')
    ax1.plot(horizon_limit_x, beam_limit_y_neg, color='black')
    ax1.set_ylim(k_parallel_plot[21:].min(), k_parallel_plot.max())
    ax2.set_ylim(delay[21:].min() * 1e9, delay.max() * 1e9)
    ax1.set_xlim(k_perp_plot.min(), k_perp_plot.max() / 1.5)
    ax2.set_xlim(k_perp_plot.min(), k_perp_plot.max() / 1.5)
    plt.savefig(name)


def plot_contour(limits, gleam, signal, name, delay, vmax=1e9, vmin=1e-6, cmap='gnuplot'):
    """
    Plot a contoured logarithmic scale color map of a signal in k-space, along with white and black contours that
    represent the horizon and beam limits, respectively.

    Parameters:
    -----------
    limits: tuple
        A tuple of the form (horizon_limit_x, horizon_limit_y, horizon_limit_y_neg, beam_limit_y, beam_limit_y_neg)
        where each element is a list of float values representing the limits of the horizon and the beam.
    gleam: tuple
        A tuple of the form (P_d_gleam, k_parallel_plot, k_perp_plot) where P_d_gleam is a 2D array containing
        the power spectrum values, and k_parallel_plot and k_perp_plot are arrays of the k values for the
        power spectrum in the parallel and perpendicular directions, respectively.
    signal: numpy array
        A 2D numpy array containing the signal values.
    name: str
        The name of the output file to be saved.
    delay: numpy array
        A 1D numpy array of delay values.
    vmax: float, optional
        The maximum value of the signal to be displayed, defaults to 1e9.
    vmin: float, optional
        The minimum value of the signal to be displayed, defaults to 1e-6.
    cmap: str, optional
        The color map to be used, defaults to 'gnuplot'.

    Returns:
    --------
    None
    """
    horizon_limit_x, horizon_limit_y, horizon_limit_y_neg, beam_limit_y, beam_limit_y_neg = limits
    P_d_gleam, k_parallel_plot, k_perp_plot = gleam[0], gleam[1], gleam[2]
    masked_P_d_gleam = np.ma.masked_equal(signal, 0.0, copy=False)

    fig, ax1 = plt.subplots()
    from matplotlib import ticker
    c = ax1.contourf(k_perp_plot[:-1], k_parallel_plot, signal, cmap=cmap, norm=LogNorm(), antialiased=True)
    cbar = fig.colorbar(c, pad=0.13)
    cbar.ax.set_yscale('log')

    ax2 = ax1.twinx()

    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax2.set_yscale('log')
    ax1.set_xlabel('$k_\perp [h Mpc^{-1}]$')
    ax1.set_ylabel('$k_\parallel [h Mpc^{-1}]$')
    ax2.set_ylabel('Log(Delay) $[ns]$')

    ax1.plot(horizon_limit_x, horizon_limit_y, color='white')
    ax1.plot(horizon_limit_x, beam_limit_y, color='black')
    ax1.plot(horizon_limit_x, horizon_limit_y_neg, color='white')
    ax1.plot(horizon_limit_x, beam_limit_y_neg, color='black')
    ax1.set_ylim(k_parallel_plot[21:].min(), k_parallel_plot.max())
    ax2.set_ylim(delay[21:].min() * 1e9, delay.max() * 1e9)
    ax1.set_xlim(k_perp_plot.min(), k_perp_plot.max() / 1.5)
    ax2.set_xlim(k_perp_plot.min(), k_perp_plot.max() / 1.5)
    plt.savefig(name)


def plot_lin(limits, gleam, signal, name, delay, vmax=1e8, vmin=1e0, cmap='gnuplot'):
    """
    Plot a linear scale color map of a signal in k-space, along with white and black contours that
    represent the horizon and beam limits, respectively.

    Parameters:
    -----------
    limits: tuple
        A tuple of the form (horizon_limit_x, horizon_limit_y, horizon_limit_y_neg, beam_limit_y, beam_limit_y_neg)
        where each element is a list of float values representing the limits of the horizon and the beam.
    gleam: tuple
        A tuple of the form (P_d_gleam, k_parallel_plot, k_perp_plot) where P_d_gleam is a 2D array containing
        the power spectrum values, and k_parallel_plot and k_perp_plot are arrays of the k values for the
        power spectrum in the parallel and perpendicular directions, respectively.
    signal: numpy array
        A 2D numpy array containing the signal values.
    name: str
        The name of the output file to be saved.
    delay: numpy array
        A 1D numpy array of delay values.
    vmax: float, optional
        The maximum value of the signal to be displayed, defaults to 1e9.
    vmin: float, optional
        The minimum value of the signal to be displayed, defaults to 1e-6.
    cmap: str, optional
        The color map to be used, defaults to 'gnuplot'.
    """

    horizon_limit_x, horizon_limit_y, horizon_limit_y_neg, beam_limit_y, beam_limit_y_neg = limits
    P_d_gleam, k_parallel_plot, k_perp_plot = gleam[0], gleam[1], gleam[2]
    masked_P_d_gleam = np.ma.masked_equal(signal, 0.0, copy=False)

    fig, ax1 = plt.subplots()

    c = ax1.pcolormesh(k_perp_plot[:-1], k_parallel_plot, signal,
                       norm=LogNorm(), cmap=cmap)

    # ax2 = ax1.twinx()

    ax1.set_ylim(k_parallel_plot.min(), k_parallel_plot.max())
    ax1.set_xlim(k_perp_plot.min(), k_perp_plot.max())
    # ax2.set_ylim(delay.min() * 1e9, delay.max() * 1e9)
    ax1.set_xlabel('$k_\perp [h Mpc^{-1}]$')
    ax1.set_ylabel('$k_\parallel [h Mpc^{-1}]$')
    # ax2.set_ylabel('Log(Delay) $[ns]$')
    fig.colorbar(c, label='$P_d$ $[mK^2(Mpc/h)^3]$')  # , pad=0.19)

    ax1.plot(horizon_limit_x, horizon_limit_y, color='white')
    ax1.plot(horizon_limit_x, beam_limit_y, color='black')
    ax1.plot(horizon_limit_x, horizon_limit_y_neg, color='white')
    ax1.plot(horizon_limit_x, beam_limit_y_neg, color='black')
    ax1.set_xlim(k_perp_plot.min(), k_perp_plot.max() / 1.5)
    plt.savefig(name)


def plot_eor(control, filepath, output_dir, min_freq, max_freq, channels, channel_bandwidth, dc_path):
    """
    Plot the Epoch of Reionization (EoR) power spectrum and window.

    Parameters
    ----------
    control : str
        Path to the control file containing the list of baselines.
    filepath : str
        Path to the directory containing the baseline data files.
    output_dir : str
        Path to the directory where output plots will be saved.
    min_freq : float
        Minimum frequency (in GHz) to plot.
    max_freq : float
        Maximum frequency (in GHz) to plot.
    channels : int
        Number of frequency channels.
    channel_bandwidth : float
        Bandwidth (in GHz) of each frequency channel.
    dc_path : str
        Path to the directory containing the comoving distance data files.

    Returns
    -------
    None

    Raises
    ------
    FileNotFoundError: If the control or data file paths are invalid.
    ValueError: If any of the frequency or bandwidth parameters are invalid.

    """
    freq_values = np.arange(min_freq * 1e9, max_freq * 1e9, channel_bandwidth * 1e9)
    freq_interval = channel_bandwidth * 1e9  # change this if channels change

    wavelength_values = 3e8 / freq_values
    z_values = 1420.0e6 / freq_values - 1
    delay_values = get_delay_times(freq_values, freq_interval)

    gleam_name1 = "gleam_all_freq_"
    gleam_name2 = "_MHz.vis"

    Dc_file = 'SKA_Power_Spectrum_and_EoR_Window/comoving/' + dc_path + '/los_comoving_distance.csv'
    Dc_values = get_Dc_values(Dc_file)

    delta_Dc_file = 'SKA_Power_Spectrum_and_EoR_Window/comoving/' + dc_path + '/delta_los_comoving_distance.csv'
    delta_Dc_values = get_delta_Dc_values(delta_Dc_file)

    # now try one channel only, can probably loop over other channels later
    time_samples = 1  # 40  # number of time samples to mod average over
    N_baselines = 29646  # 130816
    N_bins = 800

    gleam, gleam_control, gleam_residual, delays = get_Pd_avg_unfolded_binning(gleam_name1, gleam_name2, control,
                                                                               filepath,
                                                                               N_baselines, freq_values, freq_interval,
                                                                               channels, time_samples,
                                                                               Dc_values,
                                                                               delta_Dc_values,
                                                                               wavelength_values,
                                                                               z_values,
                                                                               N_bins)

    limits = get_limits(gleam, Dc_values, z_values, wavelength_values)

    plot_log(limits, gleam, gleam[0], output_dir + "/result_log.png", delays)
    plot_log(limits, gleam_control, gleam_control[0], output_dir + "/control_log.png", delays)
    plot_log(limits, gleam_residual, gleam_residual[0], output_dir + "/residual_log.png", delays)
    plot_log(limits, gleam_residual, gleam_control[0] / gleam[0], output_dir + "/ratio_log.png", delays)
