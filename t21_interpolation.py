"""
 Ts .mat files --> calculate T21 --> check whether interpolation makes sense
 --> calculate pixel_size, linear interpolate --> store in .FITS format with
 headers

"""

from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy.integrate import quad
import csv

"""
converts cubes of Ts to T21 assuming baryon overdensity to be 0 and xHI to be 1
"""


def Ts2T21(data_path, sim_name, redshift):
    Ts_filename = 'TsMat_' + str(redshift) + sim_name + '.mat'

    Ts_cube_dict = loadmat(data_path + Ts_filename)
    Ts_cube = Ts_cube_dict['Tlin']

    TR = (1 + redshift) * 2.725  # in K
    T21_cube = (27 * ((1 + redshift) / 10) ** 0.5) * (1 - TR / Ts_cube)  # in mK
    return T21_cube


# used in find_pix_size
def chi_integrand(z, omega_m, omega_lambda, omega_r):
    E_z = np.sqrt(omega_r * (1 + z) ** 4 + omega_m * (1 + z) ** 3 + omega_lambda)
    integrand = 1 / E_z
    return integrand


"""calculate pixel size in deg for an array of freqs, then save in csv file
in data_path"""


def find_pix_size(Dc_pix, N_pix, freq_values, data_path):
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
    hdu = fits.PrimaryHDU()
    header = hdu.header
    header["CTYPE1"] = "RA---SIN"
    header["CRVAL1"] = ra_deg
    header["CRPIX1"] = N_pix
    header["CDELT1"] = -pixel_size_deg
    header["CTYPE2"] = "DEC--SIN"
    header["CRVAL2"] = dec_deg
    header["CRPIX2"] = N_pix
    header["CDELT2"] = pixel_size_deg
    header["BUNIT"] = units
    header["CTYPE3"] = "Frequency MHz"
    header["CRVAL3"] = freq
    hdu = fits.PrimaryHDU(image_data, header)
    hdu.writeto(data_path + filename, overwrite=True)

    print('FITS file created for ' + str(freq) + 'MHz')


"""
Find T21 at non-integer z values by interpolating. Create .FITS files with
appropriate headers to capture the data.
"""


def T21_lin_interpolation(T21_slices, ini_z_values, final_freq_values, data_path,
                          N_pix, Dc_pix):
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

    data_interpolate_z = []
    for k in range(N_final):
        floor_z = np.floor(final_z_values[k])
        position = ini_z_values.index(floor_z)
        data = T21_slices[position] + (final_z_values[k] % 1) * gradient_in_z[position] + 2725  # CMB in mK

        str_freq = format(final_freq_values[k], ".3f")
        filename = 'Freq' + str_freq + 'MHz_interpolate_T21_slices_allall27_CMB.fits'
        create_fits(data_path, data, filename, final_freq_values[k], N_pix, pixel_size_deg[k])
        data_interpolate_z.append(data)

    return data_interpolate_z


# gets Dc and deltaDc which is used for part2
# returns in Mpc/h
def get_los_comoving_distances(freq_values, freq_sides, data_path):
    z_values = 1420 / freq_values - 1
    z_sides = 1420 / freq_sides - 1

    omega_m = 0.3156
    omega_lambda = 0.6844
    omega_r = 8e-5
    h = 0.6727  # dimensionless Hubble constant
    Dh = 3000  # in Mpc/h, Hubble distance

    # los comoving distance
    Dc = np.zeros(len(z_values))  # initialising
    for j in range(len(z_values)):
        [integral, error] = quad(chi_integrand, 0, z_values[j], args=(omega_m, omega_lambda, omega_r))
        Dc_value = Dh * integral
        Dc[j] = Dc_value

    with open(data_path + 'los_comoving_distance.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(freq_values)
        writer.writerow(Dc)

    # delta los comoving distance across freq channel
    store_Dc = np.zeros(len(z_sides))  # initialising
    for j in range(len(z_sides)):
        [integral, error] = quad(chi_integrand, 0, z_sides[j], args=(omega_m, omega_lambda, omega_r))
        Dc_value = Dh * integral
        store_Dc[j] = Dc_value

    delta_Dc = store_Dc[0:-2] - store_Dc[1:-1]

    with open(data_path + 'delta_los_comoving_distance.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(freq_sides)
        writer.writerow(delta_Dc)

    return Dc, delta_Dc


def main():
    # ----------------------------------------------------------------------------#
    data_path = './21cm_bigbox_256/'
    sim_name = '__410_1_50_1020_0.05_1_4.2_0_8_15_1_0.75_232_1_2_0_Legacy_256.mat'
    N_pix = 256  # Data cube shape
    Dc_pix = 3  # Mpc
    z_values = range(6, 22)  # redshift integer z from 21cm simulations
    N_z = len(z_values)

    """
    -to interpolate and create .fits files at, MHz
    -the np.around is to deal with floating point errors
    -note that all freq_values must be within (not inclusive) of the freq
    corresponding to z_values range
    """
    freq_values = np.around(np.arange(72, 109, 0.025), decimals=3)
    freq_sides = np.around(np.arange(72 - 0.025 / 2, 109, 0.025), decimals=4)
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
    plt.savefig("redshift_interpolation.png")
    input('If the linear interpolation in z looks good, press enter to continue.')

    data_interpolate_z = T21_lin_interpolation(T21_slices, z_values, freq_values, data_path,
                                               N_pix, Dc_pix)

    get_los_comoving_distances(freq_values, freq_sides, data_path)


if __name__ == '__main__':
    main()