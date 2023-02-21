from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy.integrate import quad
from scipy import interpolate
import csv


def Ts2T21(data_path, sim_name, redshift):
    """
    Converts cubes of Ts to T21 assuming baryon overdensity to be 0 and xHI to be 1.

    Parameters:
    -----------
    data_path: str
        path to directory containing Ts .mat files
    sim_name: str
        name of simulation (used to generate filename)
    redshift: float
        redshift value of interest

    Returns:
    --------
    T21_cube: ndarray
        Cube of T21 values in mK
    """
    Ts_filename = 'TsMat_' + str(redshift) + sim_name + '.mat'

    Ts_cube_dict = loadmat(data_path + Ts_filename)
    Ts_cube = Ts_cube_dict['Ts']

    TR = (1 + redshift) * 2.725  # in K
    T21_cube = (27 * ((1 + redshift) / 10) ** 0.5) * (1 - TR / Ts_cube)  # in mK
    return T21_cube


def chi_integrand(z, omega_m, omega_lambda, omega_r):
    """
    Helper function used in find_pix_size to calculate integrand.

    Parameters:
    -----------
    z: float
        Redshift value
    omega_m: float
        Density parameter for matter
    omega_lambda: float
        Density parameter for dark energy
    omega_r: float
        Density parameter for radiation

    Returns:
    --------
    integrand: float
        Value of the integrand at given redshift
    """
    E_z = np.sqrt(omega_r * (1 + z) ** 4 + omega_m * (1 + z) ** 3 + omega_lambda)
    integrand = 1 / E_z
    return integrand


def find_pix_size(Dc_pix, N_pix, freq_values, data_path):
    """
    Calculates pixel size in degrees for an array of frequencies, then saves the results
    to a csv file in data_path.

    Parameters:
    -----------
    Dc_pix: float
        Physical size of pixel in Mpc/h
    N_pix: int
        Number of pixels along each axis of the cube
    freq_values: ndarray
        Array of frequency values in MHz
    data_path: str
        Path to directory where pixel_size_deg.csv will be saved

    Returns:
    --------
    pixel_size_deg: ndarray
        Array of pixel sizes in degrees for each frequency value
    """
    z_values = 1420 / freq_values - 1

    omega_m = 0.3156
    omega_lambda = 0.6844
    omega_r = 8e-5
    h = 0.6727  # dimensionless Hubble constant
    Dh = 3000 / h  # in Mpc, Hubble distance

    pixel_size_deg = np.zeros(len(z_values))  # initialising
    for j in range(len(z_values)):
        # tranverse comoving distance
        [integral, error] = quad(chi_integrand, 0, z_values[j], args=(omega_m, omega_lambda, omega_r))
        Dm = Dh * integral
        delta_theta = Dc_pix / Dm  # radians
        delta_theta_deg = delta_theta * 360.0 / (2 * np.pi)
        pixel_size_deg[j] = delta_theta_deg

    # optional: save to csv
    with open(data_path + 'pixel_size_deg.csv', 'w') as file:
        writer = csv.writer(file)
        writer.writerow(freq_values)
        writer.writerow(pixel_size_deg)

    return pixel_size_deg


# produce a .FITS file with image_data and appropriate headers
def create_fits(data_path, image_data, filename, freq, N_pix, pixel_size_deg, mean,
                ra_deg=60.0, dec_deg=-30.0, units="mK"):
    """
    Create a FITS file from image data and header information.

    Parameters
    ----------
    data_path : str
        The path to the directory where the FITS file will be saved.
    image_data : ndarray
        The 2D image data to be saved in the FITS file.
    filename : str
        The name of the FITS file (including the extension).
    freq : float
        The frequency of the observation in MHz.
    N_pix : int
        The number of pixels per side of the square image.
    pixel_size_deg : float
        The size of each pixel in degrees.
    mean : float
        The mean value of the image data.
    ra_deg : float, optional
        The right ascension of the center of the image in degrees (default is 60.0).
    dec_deg : float, optional
        The declination of the center of the image in degrees (default is -30.0).
    units : str, optional
        The units of the image data (default is "mK").

    Returns
    -------
    None
        This function does not return anything, but it creates a FITS file in the specified directory.

    """
    hdu = fits.PrimaryHDU()
    header = hdu.header
    header["CTYPE1"] = "RA---SIN"
    header["CRVAL1"] = ra_deg
    header["CRPIX1"] = 1 + (N_pix / 2)
    header["CDELT1"] = -pixel_size_deg
    header["CTYPE2"] = "DEC--SIN"
    header["CRVAL2"] = dec_deg
    header["CRPIX2"] = 1 + (N_pix / 2)
    header["CDELT2"] = pixel_size_deg
    header["BUNIT"] = units
    header["CTYPE3"] = "Frequency MHz"
    header["CRVAL3"] = freq
    hdu = fits.PrimaryHDU((image_data - mean), header)
    hdu.writeto(data_path + filename, overwrite=True)

    print('FITS file created for ' + str(freq) + 'MHz)')
    print(mean)


def T21_lin_interpolation(T21_slices, ini_z_values, final_freq_values, data_path,
                          N_pix, Dc_pix, channel_bandwidth):
    """
    Compute T21 values at non-integer redshifts by interpolating the provided T21 slices
    at a set of initial redshift values, and create .FITS files with appropriate headers
    to store the interpolated data.

    Parameters:
    -----------
    T21_slices : numpy.ndarray
        A 3D numpy array containing the T21 brightness temperature slices at different
        redshift values.
    ini_z_values : numpy.ndarray
        A 1D numpy array containing the redshift values at which the T21 slices were
        computed.
    final_freq_values : numpy.ndarray
        A 1D numpy array containing the frequencies (in MHz) at which the T21 values
        are desired to be interpolated.
    data_path : str
        The directory path where the .FITS files will be saved.
    N_pix : int
        The number of pixels in each dimension of the output .FITS images.
    Dc_pix : float
        The pixel size in comoving distance (in Mpc/h).
    channel_bandwidth : float
        The bandwidth (in MHz) of each frequency channel.

    Returns:
    --------
    data_interpolate_z : numpy.ndarray
        A 3D numpy array containing the interpolated T21 values at each desired
        frequency value.
    """

    N_ini = len(ini_z_values)
    if N_ini < 2:
        print('Error: Insufficient data to perform interpolation.')
        return

    gradient_in_z = []
    for i in range(N_ini - 1):
        gradient = T21_slices[i + 1] - T21_slices[i]
        gradient_in_z.append(gradient)

    final_z_values = (1420 / final_freq_values) - 1
    N_final = len(final_z_values)

    pixel_size_deg = find_pix_size(Dc_pix, N_pix, final_freq_values, data_path)

    data_interpolate_z = np.zeros([N_final, np.array(T21_slices).shape[1], np.array(T21_slices).shape[2]])
    for i in range(np.array(T21_slices).shape[1]):
        for j in range(np.array(T21_slices).shape[2]):
            f = interpolate.interp1d(ini_z_values, np.array(T21_slices)[:, i, j], kind='cubic')
            data_interpolate_z[:, i, j] = f(final_z_values)

    mean = np.array([i.mean() for i in data_interpolate_z])

    for k in range(N_final):
        str_freq = format(final_freq_values[k], ".3f")
        filename = 'freq_' + str_freq + '_MHz_interpolate_T21_slices.fits'
        #create_fits(data_path, data_interpolate_z[k], filename, final_freq_values[k], N_pix, pixel_size_deg[k], mean[k])
        create_fits(data_path, np.tile(data_interpolate_z[k] * np.sqrt(channel_bandwidth * 1e6), (3,3))[int(N_pix/2):-int(N_pix/2), int(N_pix/2):-int(N_pix/2)], filename, final_freq_values[k], N_pix*2, pixel_size_deg[k], mean[k])

    return data_interpolate_z


def get_los_comoving_distances(freq_values, freq_sides, data_path):
    """
    Computes the line-of-sight comoving distances and the differences in comoving distance across frequency channels
    for given frequency values and frequency channel sides.

    Parameters:
    -----------
    freq_values : numpy array
        An array of the central frequency values of the frequency channels.
    freq_sides : numpy array
        An array of the frequency channel edges.
    data_path : str
        The path to the directory where the output files will be saved.

    Returns:
    --------
    Dc : numpy array
        An array of the line-of-sight comoving distances for each of the central frequency values in Mpc/h units.
    delta_Dc : numpy array
        An array of the differences in line-of-sight comoving distances across frequency channels in Mpc/h units.
    """

    # Calculate redshift values from given frequency values
    z_values = 1420 / freq_values - 1
    z_sides = 1420 / freq_sides - 1

    # Cosmological parameters
    omega_m = 0.3156
    omega_lambda = 0.6844
    omega_r = 8e-5
    h = 0.6727  # dimensionless Hubble constant
    Dh = 3000  # in Mpc/h, Hubble distance

    # Calculate line-of-sight comoving distances for each central frequency value
    Dc = np.zeros(len(z_values))  # initialize array
    for j in range(len(z_values)):
        [integral, error] = quad(chi_integrand, 0, z_values[j], args=(omega_m, omega_lambda, omega_r))
        Dc_value = Dh * integral
        Dc[j] = Dc_value

    # Save line-of-sight comoving distances to a file
    with open(data_path + 'los_comoving_distance.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(freq_values)
        writer.writerow(Dc)

    # Calculate differences in line-of-sight comoving distances across frequency channels
    store_Dc = np.zeros(len(z_sides))  # initialize array
    for j in range(len(z_sides)):
        [integral, error] = quad(chi_integrand, 0, z_sides[j], args=(omega_m, omega_lambda, omega_r))
        Dc_value = Dh * integral
        store_Dc[j] = Dc_value

    delta_Dc = store_Dc[0:-2] - store_Dc[1:-1]

    # Save differences in line-of-sight comoving distances to a file
    with open(data_path + 'delta_los_comoving_distance.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(freq_sides)
        writer.writerow(delta_Dc)

    return Dc, delta_Dc


def main():
    """
    This is the main function for a script. It executes several functions and operations to extract the 21cm signal
    from a data cube and interpolate it at a range of frequencies. It then computes the line-of-sight comoving distance
    and stores the results in files. The function also generates a plot for a quick sanity check on the interpolation.
    """
    # ----------------------------------------------------------------------------#
    data_path = 'SKA_Power_Spectrum_and_EoR_Window/comoving/test/'
    sim_name = ''
    N_pix = 256  # Data cube shape
    Dc_pix = 3  # Mpc
    z_values = range(6, 49)  # redshift integer z from 21cm simulations
    N_z = len(z_values)
    min_freq = 70
    max_freq = 110.01
    channel_bandwidth = 0.0120

    """
    -to interpolate and create .fits files at, MHz
    -the np.around is to deal with floating point errors
    -note that all freq_values must be within (not inclusive) of the freq
    corresponding to z_values range
    """
    freq_values = np.around(np.arange(min_freq, max_freq, channel_bandwidth), decimals=3)
    freq_sides = np.around(np.arange(min_freq - channel_bandwidth / 2, max_freq, channel_bandwidth), decimals=4)
    # ----------------------------------------------------------------------------#

    # initialise empty list, T21 arrays from redshifts 12 to 19 loaded to list
    T21_cubes = []
    T21_slices = []
    for i in range(N_z):
        T21 = Ts2T21(data_path, sim_name, z_values[i])
        T21_cubes.append(T21)
        T21_slices.append(T21[:, :, 27])  # random 2D slice
        print('The data cube at z=' + str(z_values[i]) + ' has shape ' + str(T21.shape))

    print('T21_cubes extracted.')

    # interpolation sanity check
    N_coords = 30  # number of points to check for in data cube
    rand_coords = np.random.randint(0, N_pix, (N_coords, 3))  # generate N_coords
    T21_at_point = np.zeros((N_z, N_coords))  # initialising
    for i in range(N_z):
        T21_cube = T21_cubes[i]
        for j in range(N_coords):
            rand_coord = rand_coords[j]
            T21_at_point[i, j] = T21_cube[rand_coord[0], rand_coord[1], rand_coord[2]]

    plt.plot(z_values, T21_at_point)
    plt.xlabel('Z')
    plt.ylabel('$T_{21}$ at random sample point')
    plt.title('21cm Cosmological Signal Per Cube Interpolation')
    plt.savefig("redshift_interpolation.png")
    # input('If the linear interpolation in z looks good, press enter to continue.')

    # Interpolate the 21cm signal at the range of frequencies
    data_interpolate_z = T21_lin_interpolation(T21_slices, z_values, freq_values, data_path,
                                               N_pix, Dc_pix, channel_bandwidth)

    # Compute the line-of-sight comoving distance and store the results in files
    get_los_comoving_distances(freq_values, freq_sides, data_path)


if __name__ == '__main__':
    main()
