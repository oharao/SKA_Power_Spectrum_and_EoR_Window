import os

import astropy.units as u
import healpy as hp
import numpy as np
import oskar
from pygdsm import HaslamSkyModel, GlobalSkyModel16

from SKA_Power_Spectrum_and_EoR_Window.sky_map.read_gleam import gleam


def gsm(date, frequency):
    # resolution defaults to 48 arcmin for freq < 10GHz
    gsm_2016 = GlobalSkyModel16(freq_unit='MHz', data_unit='MJysr', include_cmb=True)
    gsm_map = gsm_2016.generate(frequency)

    # Get the nside and number of pixels in your map
    nside = 2048
    npix = hp.nside2npix(nside)

    strperpix = (4 * np.pi) / npix

    filename = date + "_sky_maps/gsm_" + str(np.round(frequency, decimals=3)) + ".fits"
    hp.write_map(filename, gsm_2016.generated_map_data * strperpix * 1e6,
                 overwrite=True, nest=False, coord='G')
    return filename, 'Jy/pixel'


def haslam(date, frequency):
    haslam = HaslamSkyModel(freq_unit='MHz', spectral_index=-2.55, include_cmb=True)
    haslam_map = haslam.generate(frequency)
    haslam_ud_map = hp.pixelfunc.ud_grade(haslam_map, 2048)
    filename = date + "_sky_maps/gsm_" + str(np.round(frequency, decimals=3)) + ".fits"
    hp.write_map(filename, haslam_ud_map, overwrite=True, nest=False, coord='G')
    return filename, 'K'


def gaussian_broadening(sky_map, scale=1.5, nside=1024):
    sky_map_array = sky_map.to_array()
    pixel_size = (hp.nside2resol(nside, arcmin=True) * u.arcmin).to(u.arcsec)
    component_size = np.ones_like(sky_map_array[:, 0]) * pixel_size.value * scale
    sky_map_array[:, 9:11] = np.array([component_size, component_size]).T
    return oskar.Sky.from_array(sky_map_array)


def composite_map(date, frequency_hz, dc_path, eor=True, foregrounds=True, gaussian_shape=False):
    freq_name = "freq_%.3f_MHz" % (frequency_hz / 1e6)

    try:
        os.mkdir(date + '_sky_maps')
    except FileExistsError:
        pass

    composite_sky_model = oskar.Sky()

    if foregrounds is True:
        g = gleam()

        # foreground_path, units = gsm(date, frequency_hz/1e6)
        foreground_path, units = haslam(date, frequency_hz / 1e6)
        diffuse_map = oskar.Sky.from_fits_file(foreground_path, default_map_units=units,
                                               override_units=True, frequency_hz=frequency_hz)
        if gaussian_shape is True:
            diffuse_map = gaussian_broadening(diffuse_map)

        gleam_map = oskar.Sky.from_array(g.sky_array)

        composite_sky_model.append(diffuse_map)
        composite_sky_model.append(gleam_map)

        os.remove(foreground_path)

    if eor is True:
        root_path = 'SKA_Power_Spectrum_and_EoR_Window/comoving/' + dc_path

        eor_file = root_path + '/' + freq_name + '_interpolate_T21_slices.fits'
        eor_map = oskar.Sky.from_fits_file(eor_file, default_map_units='mK', frequency_hz=frequency_hz)

        composite_sky_model.append(eor_map)

    return composite_sky_model
