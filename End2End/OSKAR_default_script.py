from astropy.io import fits
from astropy.time import Time, TimeDelta
import numpy
import os
import csv
import math


def get_start_time(ra0_deg, length_sec):
    """Calculate the optimal start time for a given field RA and observation length.

    Parameters
    ----------
    ra0_deg : float
        The right ascension of the target field, in degrees.
    length_sec : float
        The length of the observation, in seconds.

    Returns
    -------
    float: The optimal start time for the observation, in UTC format.
    """
    t = Time('2000-01-01 00:00:00', scale='utc', location=('116.764d', '0d'))
    dt_hours = (24.0 - t.sidereal_time('apparent').hour) / 1.0027379
    dt_hours += (ra0_deg / 15.0)
    start = t + TimeDelta(dt_hours * 3600.0 - length_sec / 2.0, format='sec')
    return start.value


def run_oskar_gleam_model(date, min_freq, channels, channel_bandwidth, eor=False, foregrounds=True, dc_path=None):
    """Run a simulation of an interferometer telescope with a GLEAM sky model.

    Parameters
    ----------
    date : str
        The date of the observation in 'YYYY-MM-DD' format.
    min_freq : float
        The minimum frequency of the observation in GHz.
    channels : int
        The number of frequency channels to simulate.
    channel_bandwidth : float
        The bandwidth of each frequency channel in GHz.
    eor : bool, optional
        Whether to include EoR signal in the simulation (default False).
    foregrounds : bool, optional
        Whether to include foregrounds in the simulation (default True).
    dc_path : str, optional
        The path to the data cube for the EoR signal (default None).

    Returns
    -------
    None
    """
    import oskar
    # Telescope and observation parameters.
    ra0_deg = 60.0
    dec0_deg = -30.0
    length_sec = 0.0
    start_frequency_hz = min_freq * 1e9
    frequency_inc_hz = channel_bandwidth * 1e9
    num_channels = channels

    # Load sky model from GLEAM FITS binary table.
    data = fits.getdata("SKA_Power_Spectrum_and_EoR_Window/End2End/GLEAM_EGC.fits", 1)
    flux = data["int_flux_076"]
    alpha = data["alpha"]
    flux = numpy.nan_to_num(flux)
    alpha = numpy.nan_to_num(alpha)
    zeros = numpy.zeros_like(flux)
    ref_freq = 76e6 * numpy.ones_like(flux)
    sky_array = numpy.column_stack(
        (data["RAJ2000"], data["DEJ2000"],
         flux, zeros, zeros, zeros, ref_freq, alpha))

    # Create the sky model.
    sky_gleam = oskar.Sky.from_array(sky_array)

    # Create directory to write OSKAR visibilities.
    os.mkdir(date + '_vis/')

    # Loop over frequency channels.
    for c in range(num_channels):
        # Get the FITS filename.
        frequency_hz = start_frequency_hz + c * frequency_inc_hz
        freq_name = "freq_%.3f_MHz" % (frequency_hz / 1e6)
        root_name = "gleam_all_%s" % freq_name

        # Run simulation.
        params = {
            "simulator/max_sources_per_chunk": 20000,
            "simulator/write_status_to_log_file": True,
            "sky/common_flux_filter/flux_min": -1e10,
            "sky/common_flux_filter/flux_max": 1e10,
            "observation/num_channels": 1,
            "observation/start_frequency_hz": frequency_hz,
            "observation/phase_centre_ra_deg": ra0_deg,
            "observation/phase_centre_dec_deg": dec0_deg,
            "observation/num_time_steps": 1,
            "observation/start_time_utc": get_start_time(ra0_deg, length_sec),
            "observation/length": length_sec,
            "telescope/input_directory": date + '_telescope_model',
            "telescope/normalise_beams_at_phase_centre": False,
            "telescope/aperture_array/array_pattern/normalise": True,
            "telescope/aperture_array/element_pattern/normalise": True,
            "telescope/aperture_array/element_pattern/swap_xy": True,
            "interferometer/max_time_samples_per_block": 1,
            "interferometer/channel_bandwidth_hz": frequency_inc_hz,
            "interferometer/time_average_sec": 0.9,
        }

        composite_sky_model = sky_gleam.create_copy()

        if eor is True:
            root_path = 'SKA_Power_Spectrum_and_EoR_Window/comoving/' + dc_path

            eor_file = root_path + '/' + freq_name + '_interpolate_T21_slices.fits'
            sky_eor = oskar.Sky.from_fits_file(
                eor_file,
                min_peak_fraction=-1e6,
                min_abs_val=-1e6,
                frequency_hz=frequency_hz
            )
            if foregrounds is True:
                composite_sky_model.append(sky_eor)
            else:
                composite_sky_model = sky_eor.create_copy()

            pixel_size_path = root_path + '/pixel_size_deg.csv'

            with open(pixel_size_path, 'r') as file:
                reader = csv.reader(file)
                data = list(reader)
                freq_index = data[0].index('%s' % float("%.3f" % (frequency_hz / 1e6)))
                pixel_size_deg = data[1][freq_index]

            params["interferometer/uv_filter_min"] = 1.0 / math.radians(float(pixel_size_deg)*256)
            params["interferometer/uv_filter_max"] = 1.0 / (2.0*math.radians(float(pixel_size_deg)))

        settings_sim = oskar.SettingsTree("oskar_sim_interferometer")
        settings_sim.from_dict(params)

        # Run simulation.
        settings_sim["interferometer/oskar_vis_filename"] = date + '_vis/' + root_name + ".vis"
        sim = oskar.Interferometer(settings=settings_sim)
        sim.set_sky_model(composite_sky_model)
        sim.run()
        del sim
