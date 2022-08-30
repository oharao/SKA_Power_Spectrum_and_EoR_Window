from astropy.io import fits
from astropy.time import Time, TimeDelta
import numpy
import os


def get_start_time(ra0_deg, length_sec):
    """Returns optimal start time for field RA and observation length."""
    t = Time('2000-01-01 00:00:00', scale='utc', location=('116.764d', '0d'))
    dt_hours = (24.0 - t.sidereal_time('apparent').hour) / 1.0027379
    dt_hours += (ra0_deg / 15.0)
    start = t + TimeDelta(dt_hours * 3600.0 - length_sec / 2.0, format='sec')
    return start.value


def run_oskar_gleam_model(telescope_model, min_freq, channels, channel_bandwidth):
    """Main function."""
    import oskar
    # Telescope and observation parameters.
    ra0_deg = 60.0
    dec0_deg = -30.0
    length_sec = 0.0
    start_frequency_hz = min_freq * 10 ** 9
    frequency_inc_hz = channel_bandwidth * 10 ** 9
    num_channels = channels

    # Load sky model from GLEAM FITS binary table.
    data = fits.getdata("SKA_Power_Spectrum_and_EoR_Window/End-2-End/GLEAM_EGC.fits", 1)
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
    sky = oskar.Sky.from_array(sky_array)

    os.mkdir(telescope_model + '_vis/')
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
            "telescope/input_directory": telescope_model+'_telescope_model',
            "telescope/normalise_beams_at_phase_centre": False,
            "telescope/aperture_array/array_pattern/normalise": True,
            "telescope/aperture_array/element_pattern/normalise": True,
            "telescope/aperture_array/element_pattern/swap_xy": True,
            "interferometer/max_time_samples_per_block": 1,
            "interferometer/channel_bandwidth_hz": frequency_inc_hz,
            "interferometer/time_average_sec": 0.9,
        }
        settings_sim = oskar.SettingsTree("oskar_sim_interferometer")
        settings_sim.from_dict(params)

        # Run simulation.
        settings_sim["interferometer/ms_filename"] = telescope_model + '_vis/' + root_name + ".ms"
        sim = oskar.Interferometer(settings=settings_sim)
        sim.set_sky_model(sky)
        sim.run()
        del sim
