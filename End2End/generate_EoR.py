"""
blocks means binning
reduced refers to attempt at averging abs(v^2) over the +ve/-ve delay
"""

import csv
from bisect import bisect_left

import astropy.constants as const
import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import oskar
from astropy.cosmology import FlatLambdaCDM
from matplotlib.colors import LogNorm
from scipy.fft import fft, fftfreq
from scipy.signal import windows


# take in freq in Hz
def delay_transform(name1, name2, filepath, N, freq_values, channels, baseline_mag, control=False):
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
            vis_data[k, :] = vis[0, 0, :, 0][baseline_mag.argsort()]

    # Account for the summation across time samples.
    vis_data = vis_data / header.num_blocks

    window_vector = windows.blackmanharris(channels) ** 2
    window_array = np.tile(window_vector, (len(vis_data[0]), 1)).T
    window_norm = np.sum(window_vector)
    vis_data = vis_data * window_array / window_norm
    vis_delay = fft(vis_data, axis=0)  # normalisation by 1/N because norm=forward/backwards does not work
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

    print(filepath + "/" + name1 + freq + name2)
    (header, handle) = oskar.VisHeader.read(filepath + "/" + name1 + freq + name2)
    block = oskar.VisBlock.create_from_header(header)
    for i in range(header.num_blocks):
        block.read(header, handle, i)
        u = block.baseline_uu_metres()
        v = block.baseline_vv_metres()
        w = block.baseline_ww_metres()
    uvw_data = np.array((u, v, w))  # structure: uvw
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
    delay_time = fftfreq(len(freq), freq_interval)  # /(2*np.pi)
    # delay_time = 1/(freq - freq[0] + freq_interval)
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
    H0 = 70  # km/s/Mpc
    f_21 = (1420.0 * u.MHz).to(1 / u.s)
    cosmo = FlatLambdaCDM(H0=H0, Om0=omega_m)

    # Calculate k_parallel
    k_parallel = (tau_vec * np.divide(2 * np.pi * cosmo.H(z) * f_21, const.c.to(u.km / u.s) * (1 + z) ** 2))

    return k_parallel.value


def get_k_perp(baseline_mag_vec, freq):
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
    # Set cosmological parameters
    omega_m = 0.3156
    H0 = 70  # km/s/Mpc
    f_21 = (1420.0 * u.MHz).to(1 / u.s)
    cosmo = FlatLambdaCDM(H0=H0, Om0=omega_m)

    # Calculate k_perp using the given inputs
    z = f_21 / (freq * u.Hz) - 1
    wavelength = const.c / freq
    k_perp = np.divide(2 * np.pi * (baseline_mag_vec / wavelength), cosmo.comoving_transverse_distance(z))

    return k_perp.value


def get_cosmological_model(H0=100, Omega_m0=0.3156):
    from astropy.cosmology import FlatLambdaCDM
    cosmo = FlatLambdaCDM(H0=H0, Om0=Omega_m0)
    return cosmo


def get_pixel_size(z, model, pixel=3 * u.Mpc):
    return (model.arcsec_per_kpc_comoving(z) * pixel.to(u.kpc)).to(u.deg)


def get_delta_dc(freq, bandwidth, model):
    z = 1420.0e6 / freq - 1
    z_interval = z - (1420.0e6 / (freq + bandwidth) - 1)
    delta_dc = np.abs(
        model.comoving_transverse_distance(z[:, None] + z_interval[:, None]) - model.comoving_transverse_distance(
            z[:, None]))
    return delta_dc


def get_dc(freq, model):
    z = 1420.0e6 / freq - 1
    dc = model.comoving_transverse_distance(z)
    return dc


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
    scale_factor = 1.5

    # Calculate block boundaries using power law binning
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


def get_Pd_avg_unfolded_binning(name1: str, name2: str, filepath: str,
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
    vis_position, baseline_block_boundaries = get_vis_boundaries(sorted_baseline_mag, N_bins)

    sum_sorted_vis = np.zeros([len(sorted_delay_values), N_bins])

    delay_data = delay_transform(name1, name2, filepath, len(baseline_mag), freq_values,
                                 channels, baseline_mag)

    vis_delay = np.abs(delay_data) ** 2  # get modulus squared

    # sort by delay values
    sorted_vis_delay = vis_delay[delay_values.argsort(), :]

    sorted_vis_delay_bins = np.zeros([channels, N_bins])
    for q in range(N_bins):
        if vis_position[q] != vis_position[q + 1]:
            sorted_vis_delay_bins[:, q] = np.sum(sorted_vis_delay[:, vis_position[q]:vis_position[q + 1] - 1],
                                                 axis=-1) / (vis_position[q + 1] - vis_position[q])

        sum_sorted_vis = sum_sorted_vis + sorted_vis_delay_bins

    avg_sorted_vis = sum_sorted_vis / time_samples

    k_parallel = get_k_parallel(z[delay_values.argsort()], sorted_delay_values)
    k_perp = get_k_perp(baseline_block_boundaries, np.median(freq_values))

    k_B = 1.380649e-23  # Boltzman constant

    A_eff = np.pi * (38 * u.m / 2) ** 2
    factor = ((1 * u.Jy).to(u.J / (u.m ** 2))) ** 2 * (
                (np.divide((wavelength * u.m) ** 2, 2 * const.k_B.to(u.J / u.mK))) ** 2) * np.divide(
        (Dc ** 2 * delta_Dc.T)[0], freq_interval * u.Hz) * np.divide(A_eff,
                                                                     (wavelength * u.m) ** 2 * freq_interval * u.Hz)

    P_d = avg_sorted_vis * factor[:, None] * freq_interval ** 2

    xx, yy = np.meshgrid(k_perp[:-1], k_parallel)
    k = np.sqrt(yy ** 2 + xx ** 2)

    delta = P_d  # * k ** 3 / (2 * np.pi**2)

    eor = delta.value, k_parallel, k_perp[:-1]

    return eor, sorted_delay_values, baseline_block_boundaries


def get_limits(signal, Dc_values, z_values, wavelength_values, fov_angle):
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
    cosmo = get_cosmological_model()
    k_mode_gradient = np.divide(Dc_values * cosmo.H(z_values), const.c.to(u.km / u.s) * (1 + z_values)).value.max()
    horizon_limit_gradient = np.sin(np.radians(fov_angle)) * k_mode_gradient
    horizon_limit_x = k_perp_plot

    # calculate horizon limit and beam limit y values
    horizon_limit_y = horizon_limit_x * horizon_limit_gradient
    beam_limit_gradient = k_mode_gradient * np.sin((wavelength_values/38).max())
    beam_limit_y = beam_limit_gradient * horizon_limit_x
    horizon_limit_y_neg = -horizon_limit_x * horizon_limit_gradient
    beam_limit_y_neg = -beam_limit_gradient * horizon_limit_x

    # return limit values
    return horizon_limit_x, horizon_limit_y, horizon_limit_y_neg, beam_limit_y, beam_limit_y_neg


def plot_lognorm(limits, gleam, name, delay, baselines, vmax=1e14, vmin=1e-8, cmap='gnuplot'):
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
    masked_P_d_gleam = np.ma.masked_equal(P_d_gleam, 0.0, copy=False)

    shift = 0  # K_par axis shift
    k_par_index = int(len(k_parallel_plot) / 2) + 1 + shift

    k_par_slice = int(5 * len(k_parallel_plot) / 9)
    k_perp_slice = int(len(k_perp_plot) / 2)


    fig = plt.figure(figsize=(11, 10))
    ax1 = plt.subplot2grid((12, 10), (2, 0), rowspan=10, colspan=6)

    pcm = ax1.pcolormesh(k_perp_plot, k_parallel_plot,
                         P_d_gleam, cmap=cmap, norm=LogNorm(vmin=vmin, vmax=vmax))
    ax1.set_xlim(k_perp_plot.min(), k_perp_plot.max() / 1.3)
    ax1.set_ylim(k_parallel_plot[k_par_index], k_parallel_plot.max())

    # Set the scales of the axes to log
    ax1.set_xscale('log')
    ax1.set_yscale('log')

    # Add labels to the axes
    ax1.set_xlabel('$k_\perp [h Mpc^{-1}]$')
    ax1.set_ylabel('$k_\parallel [h Mpc^{-1}]$')

    ax1.plot(horizon_limit_x, horizon_limit_y, color='black')
    ax1.plot(horizon_limit_x, horizon_limit_y + 0.1, color='black', linestyle='dotted')
    ax1.plot(horizon_limit_x, beam_limit_y, color='black', linestyle='dashed')
    ax1.plot(horizon_limit_x, horizon_limit_y_neg, color='black')
    ax1.plot(horizon_limit_x, beam_limit_y_neg, color='black', linestyle='dashed')

    ax1.vlines(k_perp_plot[k_perp_slice], k_parallel_plot.min(), k_parallel_plot.max(), color='blue')
    ax1.hlines(k_parallel_plot[k_par_slice], k_perp_plot.min(), k_perp_plot.max(), color='red')

    # Add k_perp slice
    ax2 = plt.subplot2grid((12, 10), (0, 0), rowspan=2, colspan=6)
    ax2.loglog(k_perp_plot, P_d_gleam[k_par_slice, :], 'red')
    ax2.set_ylabel('$P(k)$ $[mK^2 Mpc^3]$')
    ax2.xaxis.set_ticks_position('none')
    ax2.set(xticklabels=[])
    ax2.set_xlim(k_perp_plot.min(), k_perp_plot.max() / 1.3)
    ax4 = ax2.twiny()
    ax4.set_xscale('log')
    ax4.set_xlabel('Baseline Length ($\lambda$)')
    ax4.set_xlim(baselines.min(), baselines.max() / 1.3)

    # Add K_parallel slice
    ax3 = plt.subplot2grid((12, 10), (2, 6), rowspan=10, colspan=2)
    ax3.loglog(P_d_gleam[:, k_perp_slice], k_parallel_plot, 'blue')
    ax3.set_xlabel('$P(k)$ $[mK^2 Mpc^3]$')
    ax3.yaxis.set_ticks_position('none')
    ax3.set(yticklabels=[])
    ax3.set_ylim(k_parallel_plot[k_par_index], k_parallel_plot.max())
    ax5 = ax3.twinx()
    ax5.set_yscale('log')
    ax5.set_ylabel('Delay ($ns$)')
    ax5.set_ylim(delay[k_par_index].min(), delay.max() / 1.3)

    ax2.xaxis.grid(True, which='major', linestyle='dashed')
    ax3.yaxis.grid(True, which='major', linestyle='dashed')
    ax1.grid(True, which='major', linestyle='dashed')

    ax3.hlines(horizon_limit_y[k_perp_slice], P_d_gleam[:, k_perp_slice].min(),
               P_d_gleam[:, k_perp_slice].max(), color='black')
    ax3.hlines(beam_limit_y[k_perp_slice], P_d_gleam[:, k_perp_slice].min(),
               P_d_gleam[:, k_perp_slice].max(), color='black', linestyle='dashed')
    ax2.vlines(k_perp_plot[np.argmax((horizon_limit_y >= k_parallel_plot[k_par_slice]) == True)],
               P_d_gleam[k_par_slice, :].min(),
               P_d_gleam[k_par_slice, :].max(), color='black')

    plt.subplots_adjust(hspace=0.05, wspace=0.05)

    cbar_ax = fig.add_axes([0.1257, 0.03, 0.463, 0.02])
    # Add a colorbar
    cbar = fig.colorbar(pcm, cax=cbar_ax, orientation='horizontal', label='$P(k)$ $[mK^2 Mpc^3]$', extend='both')

    plt.savefig(name, bbox_inches='tight')


def plot_eor(filepath, output_dir, min_freq, max_freq, channels, channel_bandwidth, observation_num_time_steps):
    """
    Plot the Epoch of Reionization (EoR) power spectrum and window.

    Parameters
    ----------
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
    fov_angle = 90 # 86.8  # in Degrees

    freq_values = np.arange(min_freq * 1e9, max_freq * 1e9, channel_bandwidth * 1e9)
    freq_interval = channel_bandwidth * 1e9  # change this if channels change

    wavelength_values = 3e8 / freq_values
    z_values = 1420.0e6 / freq_values - 1

    gleam_name1 = "gleam_all_freq_"
    gleam_name2 = "_MHz.vis"

    cosmo = get_cosmological_model()
    Dc_values = get_dc(freq=freq_values, model=cosmo).value
    delta_Dc_values = get_delta_dc(freq=freq_values, bandwidth=freq_interval, model=cosmo).value

    N_bins = 250

    gleam, delays, baselines = get_Pd_avg_unfolded_binning(gleam_name1, gleam_name2, filepath,
                                                           freq_values, freq_interval,
                                                           channels, observation_num_time_steps, Dc_values,
                                                           delta_Dc_values, wavelength_values, z_values,
                                                           N_bins)

    limits = get_limits(gleam, Dc_values, z_values, wavelength_values, fov_angle)

    baselines_wavelengths = baselines / np.median(wavelength_values)

    plot_lognorm(limits, gleam, output_dir + "/result_log.png", delays * 1e9, baselines_wavelengths)
    np.save(output_dir + '/Delta_PS.npy', np.array(gleam, dtype=object))

    return limits, output_dir + '/Delta_PS.npy', output_dir, delays * 1e9, baselines_wavelengths
