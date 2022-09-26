import matplotlib.pyplot as plt
from astropy.visualization import astropy_mpl_style
plt.style.use(astropy_mpl_style)

from astropy.utils.data import get_pkg_data_filename
from astropy.io import fits
import numpy as np

image_file = 'image_plot/100MHz_big_I.fits'

image_data = fits.getdata(image_file, ext=0)

plt.figure()
plt.pcolormesh(np.linspace(-768, 768, 1536), np.linspace(-768, 768, 1536), image_data[0], cmap='gray')
#plt.pcolormesh(np.linspace(-128, 128, 256), np.linspace(-128, 128, 256), image_data[0], cmap='gray')
plt.colorbar()
#plt.vlines(-2, -2, 2)
#plt.vlines(2, -2, 2)
#plt.hlines(-2, -2, 2)
#plt.hlines(2, -2, 2)
plt.savefig('image_plot/100MHz_I.png')