import numpy as np
import pandas as pd
import healpy as hp
import pickle
import matplotlib.pyplot as plt
from astropy.io import fits


class gleam(object):
    def __init__(self):
        self.df = fits.getdata("SKA_Power_Spectrum_and_EoR_Window/sky_map/GLEAM_EGC_v2.fits", 1)
        self.RAdeg = self.df['RAJ2000']
        self.DEdeg = self.df['DEJ2000']
        self.flux = np.nan_to_num(self.df['int_flux_076'])
        self.alpha = np.nan_to_num(self.df['alpha'])
        self.zeros = np.zeros_like(self.flux)
        self.ref_freq = 76e6 * np.ones_like(self.flux)
        self.sky_array = np.column_stack(
            (self.RAdeg, self.DEdeg, self.flux, self.zeros, self.zeros, self.zeros, self.ref_freq, self.alpha))


'''
    def generate(self, freq):
        delta_freq = np.log10(freq / 76)
        colors = np.nan_to_num(self.df['int_flux_076'])
        new_colors = []
        for i in range(len(colors)):
            if self.alpha[i] != '---':
                if colors[i] != '---':
                    new_colors.append(10 ** ((delta_freq * float(self.alpha[i])) + np.log10(float(colors[i]))))
            else:
                new_colors.append('---')
        new_colors = np.nan_to_num(np.array(new_colors))

        nside = 1024
        npix = hp.nside2npix(nside)

        # Go from HEALPix coordinates to indices

        theta = -np.deg2rad(self.DEdeg) + 0.5 * np.pi

        indices = hp.ang2pix(nside, theta, np.deg2rad(self.RAdeg))

        # Initate the map and fill it with the values
        hpxmap = np.zeros(npix)
        for i in range(len(new_colors)):
            if type(colors[i]) is not str:
                hpxmap[indices[i]] = new_colors[i]

        return self.change_coord(hpxmap, ['C', 'G'])

    def change_coord(self, m, coord):
        """ Change coordinates of a HEALPIX map

        Parameters
        ----------
        m : map or array of maps
          map(s) to be rotated
        coord : sequence of two character
          First character is the coordinate system of m, second character
          is the coordinate system of the output map. As in HEALPIX, allowed
          coordinate systems are 'G' (galactic), 'E' (ecliptic) or 'C' (equatorial)

        Example
        -------
        The following rotate m from galactic to equatorial coordinates.
        Notice that m can contain both temperature and polarization.
        >>>> change_coord(m, ['G', 'C'])
        """
        # Basic HEALPix parameters
        npix = m.shape[-1]
        nside = hp.npix2nside(npix)
        ang = hp.pix2ang(nside, np.arange(npix))

        # Select the coordinate transformation
        rot = hp.Rotator(coord=reversed(coord))

        # Convert the coordinates
        new_ang = rot(*ang)
        new_pix = hp.ang2pix(nside, *new_ang)

        return m[..., new_pix]
'''
