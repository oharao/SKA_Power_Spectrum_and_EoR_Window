import matplotlib.pyplot as plt
from astropy.visualization import astropy_mpl_style
plt.style.use(astropy_mpl_style)

from astropy.utils.data import get_pkg_data_filename
from astropy.io import fits
import numpy as np
from matplotlib.colors import LogNorm, SymLogNorm
import math
import astropy.constants as const


image_file = 'test_I.fits'
freq = 130
fov = 5

image_data = fits.getdata(image_file, ext=0)
beam_theta = math.degrees(np.divide(const.c.value,freq*1e6)/38)


fig, ax = plt.subplots()
eor = ax.pcolormesh(np.linspace(-fov/2, fov/2, 1024), np.linspace(-fov/2, fov/2, 1024), image_data[0], cmap='gray')
#plt.pcolormesh(np.linspace(-128, 128, 256), np.linspace(-128, 128, 256), image_data[0], cmap='gray')
circle = plt.Circle((0, 0), beam_theta/2, color='b', fill=False)
ax.add_patch(circle)
cbar = fig.colorbar(eor)
plt.savefig(image_file[:-5] + '.png')