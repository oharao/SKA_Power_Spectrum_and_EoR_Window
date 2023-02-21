import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy import interpolate
import csv
from scipy.io import loadmat
from scipy.fft import fft, fftfreq
import scipy.ndimage as ndimage
from scipy.signal.windows import blackmanharris
import os
import astropy.units as u
import astropy.constants as const
from astropy.io import fits

# We change the default level of the logger so that
# we can see what's happening with caching.
import logging, sys, os

logger = logging.getLogger('21cmFAST')
logger.setLevel(logging.INFO)

import py21cmfast as p21c
import tools21cm as t2c
# For plotting the cubes, we use the plotting submodule:
from py21cmfast import plotting
# For interacting with the cache
from py21cmfast import cache_tools

print(f"Using 21cmFAST version {p21c.__version__}")

cache_path = 'SKA_Power_Spectrum_and_EoR_Window/End2End/_cache'
if not os.path.exists(cache_path):
    os.mkdir(cache_path)

p21c.config['direc'] = cache_path
cache_tools.clear_cache(direc=cache_path)

#############################################################


def create_coeval_cubes(z_values, cube_len_in_pixels=100, pixel_size_cMpc=3):
    coeval_cubes = p21c.run_coeval(redshift=list(z_values),
                                   user_params={"HII_DIM": cube_len_in_pixels,
                                                "BOX_LEN": cube_len_in_pixels*pixel_size_cMpc,
                                                "USE_INTERPOLATION_TABLES": True},
                                   cosmo_params=p21c.CosmoParams(SIGMA_8=0.8),
                                   astro_params=p21c.AstroParams({"HII_EFF_FACTOR": 20.0}),
                                   random_seed=12345)
    return np.array(coeval_cubes)


def create_lightcone(coeval_list):
    cube_size_deg = np.array(
        [[t2c.cm.angular_size_comoving(i.user_params.BOX_LEN, i.redshift), i.user_params.HII_DIM, i.redshift] for i in
         coeval_list])
    lightcone = np.array([i.brightness_temp[:, :, 27] for i in coeval_list])

    beam_size_max = np.divide(const.c.value / 1420e6 * (cube_size_deg[:, 2].max() + 1), 38) * u.rad.to(u.deg)
    tile_pixels = np.ceil(beam_size_max / cube_size_deg[:, 0].min() * cube_size_deg[:, 1].max()) // 2 * 2 + 1
    lightcone_padded = t2c.padding_lightcone(lightcone.T, int(np.ceil(tile_pixels / 2))).T
    return lightcone_padded, cube_size_deg


def create_eor_fits(lightcone, cube_size_deg, path, ra_deg, dec_deg):
    if not os.path.exists(path):
        os.mkdir(path)
    for i, cube_stats in zip(lightcone, cube_size_deg):
        str_freq = format(t2c.z_to_nu(cube_stats[2]), ".3f")
        filename = 'freq_' + str_freq + '_MHz_interpolate_T21_slices.fits'

        hdu = fits.PrimaryHDU(i)
        hdul = fits.HDUList([hdu])

        hdul[0].header.set('CTYPE1', 'RA---SIN')
        hdul[0].header.set('CTYPE2', 'DEC--SIN')
        hdul[0].header.set('CTYPE3', 'FREQ')
        hdul[0].header.set('CRVAL1', ra_deg)
        hdul[0].header.set('CRVAL2', dec_deg)
        hdul[0].header.set('CRVAL3', t2c.z_to_nu(cube_stats[2]) * 1e6)
        hdul[0].header.set('CRPIX1', 1 + i.shape[0] // 2)
        hdul[0].header.set('CRPIX2', 1 + i.shape[1] // 2)
        hdul[0].header.set('CDELT1', -cube_stats[0] / cube_stats[1])
        hdul[0].header.set('CDELT2', cube_stats[0] / cube_stats[1])
        hdul[0].header.set('BUNIT', 'mK')

        logger.info(f"Writing output lightcone to {path + filename}")
        hdul.writeto(path + filename, overwrite=True)


N_pix = 512  # Data cube
min_freq = 130
max_freq = 170
channel_bandwidth = 0.12

freq_values = np.around(np.arange(min_freq, max_freq, channel_bandwidth), decimals=3)
freq_sides = np.around(np.arange(min_freq - channel_bandwidth / 2, max_freq, channel_bandwidth), decimals=4)
z_values = 1420 / freq_values - 1

coeval = create_coeval_cubes(z_values)
lightcone, cube_size_deg = create_lightcone(coeval, baseline_max=10e3)
create_eor_fits(lightcone, cube_size_deg, 'SKA_Power_Spectrum_and_EoR_Window/comoving/21cmfast/', 60.0, -30.0)
