import csv

import astropy.units as u
import numpy as np
import scipy
import tools21cm as t2c
from astropy.io import fits
from matplotlib.transforms import (
    Bbox, TransformedBbox)
from mpl_toolkits.axes_grid1.inset_locator import (
    BboxPatch, BboxConnector, BboxConnectorPatch)
from scipy import interpolate
from scipy.integrate import quad
from scipy.io import loadmat

from End2End.generate_EoR import get_cosmological_model, get_delta_dc, get_pixel_size


def connect_bbox(bbox1, bbox2,
                 loc1a, loc2a, loc1b, loc2b,
                 prop_lines, prop_patches=None):
    if prop_patches is None:
        prop_patches = {
            **prop_lines,
            "alpha": prop_lines.get("alpha", 1) * 0.2,
            "clip_on": False,
        }

    c1 = BboxConnector(
        bbox1, bbox2, loc1=loc1a, loc2=loc2a, clip_on=False, **prop_lines)
    c2 = BboxConnector(
        bbox1, bbox2, loc1=loc1b, loc2=loc2b, clip_on=False, **prop_lines)

    bbox_patch1 = BboxPatch(bbox1, **prop_patches)
    bbox_patch2 = BboxPatch(bbox2, **prop_patches)

    p = BboxConnectorPatch(bbox1, bbox2,
                           loc1a=loc1a, loc2a=loc2a, loc1b=loc1b, loc2b=loc2b,
                           clip_on=False,
                           **prop_patches)

    return c1, c2, bbox_patch1, bbox_patch2, p


def zoom_effect01(ax1, ax2, xmin1, xmax1, xmin2, xmax2, **kwargs):
    """
    Connect *ax1* and *ax2*. The *xmin*-to-*xmax* range in both axes will
    be marked.

    Parameters
    ----------
    ax1
        The main axes.
    ax2
        The zoomed axes.
    xmin, xmax
        The limits of the colored area in both plot axes.
    **kwargs
        Arguments passed to the patch constructor.
    """

    bbox1 = Bbox.from_extents(xmin1, 0, xmax1, 1)
    bbox2 = Bbox.from_extents(xmin2, 0, xmax2, 1)

    mybbox1 = TransformedBbox(bbox1, ax1.get_xaxis_transform())
    mybbox2 = TransformedBbox(bbox2, ax2.get_xaxis_transform())

    prop_patches = {**kwargs, "ec": "none", "alpha": 0.2}

    c1, c2, bbox_patch1, bbox_patch2, p = connect_bbox(
        mybbox1, mybbox2,
        loc1a=3, loc2a=2, loc1b=4, loc2b=1,
        prop_lines=kwargs, prop_patches=prop_patches)

    # ax1.add_patch(bbox_patch1)
    # ax2.add_patch(bbox_patch2)
    ax2.add_patch(c1)
    ax2.add_patch(c2)
    ax2.add_patch(p)

    return c1, c2, bbox_patch1, bbox_patch2, p


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
def create_fits(data_path, image_data, filename, freq, N_pix, pixel_size_deg,
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
    header["CTYPE3"] = "FREQ"
    header["CRVAL3"] = freq * 1e6  # Convert to Hz
    hdu = fits.PrimaryHDU(image_data, header)
    hdu.writeto(data_path + filename, overwrite=True)

    print('FITS file created for ' + str(freq) + 'MHz)')


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


def get_cube_data(data_path, redshift):
    filename = f'{data_path}T21_cube_{str(redshift)}_Npix512.mat'
    cube_dict = loadmat(filename)
    cube_brightness_temp = cube_dict['Tlin']

    return cube_brightness_temp


def create_lightcone(data_path, freq_values_mhz, bandwidth_mhz, N_pix, pixel_size_mpc):
    cosmo = get_cosmological_model()

    z_values = t2c.nu_to_z(freq_values_mhz)

    z_open = None
    lightcone, lightcone_z = [], []
    index = []
    # obtain trend deviation caused by lightcone effect
    for z_file, i in zip(np.array(z_values, dtype=int), range(len(z_values))):
        if z_open != z_file:  # loop ensuring each file is opened once
            cube_data = get_cube_data(data_path, z_file)
            z_open = z_file

        delta_dc = get_delta_dc(np.array([freq_values_mhz.min()]) * 1e6, bandwidth_mhz * 1e6 * i, model=cosmo)[
            0, 0].value

        if (int(delta_dc / pixel_size_mpc) % N_pix) != index:  # append only unique slices and z for interp
            index = (int(delta_dc / pixel_size_mpc) % N_pix)
            lightcone.append(cube_data[:, :, index] - np.mean(cube_data))
            lightcone_z.append(z_values[i])

    # obtain trend
    global_signal = []
    z_file = np.arange(np.ceil(z_values.max()), np.floor(z_values.min()) - 1, -1, dtype=int)
    for z in z_file:
        global_signal.append(np.mean(get_cube_data(data_path, z)))
    f = scipy.interpolate.interp1d(z_file, global_signal, kind='quadratic')
    lightcone += f(lightcone_z)[:, None, None]

    lightcone_hrez_z = np.zeros([len(z_values), N_pix, N_pix])
    for i in range(np.array(lightcone).shape[1]):
        for j in range(np.array(lightcone).shape[2]):
            f = interpolate.interp1d(lightcone_z, np.array(lightcone)[:, i, j], kind='quadratic',
                                     fill_value="extrapolate")
            lightcone_hrez_z[:, i, j] = f(z_values)
            print('x: ' + str(i), 'y: ' + str(j))

    for k in range(len(z_values)):
        str_freq = format(t2c.z_to_nu(z_values[k]), ".3f")
        filename = 'freq_' + str_freq + '_MHz_interpolate_T21_slices.fits'
        create_fits(data_path, lightcone_hrez_z[k], filename, t2c.z_to_nu(z_values[k]), N_pix,
                    get_pixel_size(z_values[k], cosmo, pixel_size_mpc * u.Mpc).value)
    return lightcone_hrez_z


def main():
    """
    This is the main function for a script. It executes several functions and operations to extract the 21cm signal
    from a data cube and interpolate it at a range of frequencies. It then computes the line-of-sight comoving distance
    and stores the results in files. The function also generates a plot for a quick sanity check on the interpolation.
    """
    # ----------------------------------------------------------------------------#
    data_path = 'SKA_Power_Spectrum_and_EoR_Window/comoving/130-170MHz_512/'
    N_pix = 512  # Data cube shape
    Dc_pix = 3  # Mpc
    min_freq = 135
    max_freq = 165
    channel_bandwidth = 0.125

    """
    -to interpolate and create .fits files at, MHz
    -the np.around is to deal with floating point errors
    -note that all freq_values must be within (not inclusive) of the freq
    corresponding to z_values range
    """
    freq_values = np.around(np.arange(min_freq, max_freq, channel_bandwidth), decimals=3)
    freq_sides = np.around(np.arange(min_freq - channel_bandwidth / 2, max_freq, channel_bandwidth), decimals=4)
    # ----------------------------------------------------------------------------#

    # Interpolate the 21cm signal at the range of frequencies
    data_interpolate_z = create_lightcone(data_path=data_path, freq_values_mhz=freq_values,
                                          bandwidth_mhz=channel_bandwidth, N_pix=N_pix, pixel_size_mpc=Dc_pix)

    # Compute the line-of-sight comoving distance and store the results in files
    get_los_comoving_distances(freq_values, freq_sides, data_path)


if __name__ == '__main__':
    main()
