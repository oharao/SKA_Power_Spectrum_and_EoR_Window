import math
import os

from astropy.time import Time, TimeDelta

from SKA_Power_Spectrum_and_EoR_Window.End2End.generate_EoR import get_cosmological_model, get_pixel_size
from SKA_Power_Spectrum_and_EoR_Window.End2End.sky_map.sky_model import composite_map


def get_start_time(ra0_deg, observation_start_time_utc, observation_length_sec):
    """Calculate the optimal start time for a given field RA and observation length.

    Parameters
    ----------
    ra0_deg : float
        The right ascension of the target field, in degrees.
    observation_length_sec : float
        The length of the observation, in seconds.

    Returns
    -------
    float: The optimal start time for the observation, in UTC format.
    """
    t = Time(observation_start_time_utc, scale='utc', location=('116.764d', '0d'))
    dt_hours = (24.0 - t.sidereal_time('apparent').hour) / 1.0027379
    dt_hours += (ra0_deg / 15.0)
    start = t + TimeDelta(dt_hours * 3600.0 - observation_length_sec / 2.0, format='sec')
    return start.value


def run_oskar(date, min_freq, channels, channel_bandwidth, ra0_deg, dec0_deg, observation_start_time_utc,
              observation_length_sec, observation_num_time_steps, eor, foregrounds, gaussian_shape,
              dc_path, oskar_binary):
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
    eor : bool
        Whether to include EoR signal in the simulation (default False).
    foregrounds : bool
        Whether to include foregrounds in the simulation (default True).
    dc_path : str
        The path to the data cube for the EoR signal (default None).

    Returns
    -------
    None
    """
    import oskar
    # Telescope and observation parameters.

    start_frequency_hz = min_freq * 1e9
    frequency_inc_hz = channel_bandwidth * 1e9
    num_channels = channels

    # Create directory to write OSKAR visibilities.
    os.mkdir(date + '_vis/')

    # Loop over frequency channels.
    for c in range(num_channels):
        # Get the FITS filename.
        frequency_hz = start_frequency_hz + c * frequency_inc_hz
        freq_name = "freq_%.3f_MHz" % (frequency_hz / 1e6)
        root_name = "gleam_all_%s" % freq_name

        composite_sky_model = composite_map(date, frequency_hz, dc_path, eor, foregrounds, gaussian_shape)

        # Run simulation.
        params = {
            "simulator/max_sources_per_chunk": 500000,
            "simulator/write_status_to_log_file": True,
            "sky/common_flux_filter/flux_min": -1e10,
            "sky/common_flux_filter/flux_max": 1e10,
            "observation/num_channels": 1,
            "observation/start_frequency_hz": frequency_hz,
            "observation/phase_centre_ra_deg": ra0_deg,
            "observation/phase_centre_dec_deg": dec0_deg,
            "observation/num_time_steps": observation_num_time_steps,
            "observation/start_time_utc": get_start_time(ra0_deg, observation_start_time_utc, observation_length_sec),
            "observation/length": observation_length_sec,
            "telescope/input_directory": date + '_telescope_model',
            "telescope/normalise_beams_at_phase_centre": False,
            "telescope/aperture_array/array_pattern/normalise": True,
            "telescope/aperture_array/element_pattern/normalise": True,
            "telescope/aperture_array/element_pattern/swap_xy": True,
            "interferometer/max_time_samples_per_block": 1,
            "interferometer/channel_bandwidth_hz": frequency_inc_hz,
            "interferometer/time_average_sec": 0.9,
        }

        if eor is True:
            cosmo = get_cosmological_model()
            pixel_size_deg = get_pixel_size(z=(1420.0e6 / (frequency_hz / 1e6) - 1), model=cosmo).value

            params["interferometer/uv_filter_min"] = 1.0 / math.radians(float(pixel_size_deg) * 512)
            params["interferometer/uv_filter_max"] = 1.0 / (2.0 * math.radians(float(pixel_size_deg)))

        settings_sim = oskar.SettingsTree("oskar_sim_interferometer")
        settings_sim.from_dict(params)

        # Run simulation.
        if oskar_binary is True:
            settings_sim["interferometer/oskar_vis_filename"] = date + '_vis/' + root_name + ".vis"
        else:
            settings_sim["interferometer/ms_filename"] = date + '_vis/' + root_name + ".ms"
        sim = oskar.Interferometer(settings=settings_sim)
        sim.set_sky_model(composite_sky_model)
        sim.run()
        del sim
