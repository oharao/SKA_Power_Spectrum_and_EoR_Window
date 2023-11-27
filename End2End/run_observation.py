"""
The functions within intialise an observation using the outlined End-to-End Simulation Pipeline for SKA-LOW. This
piepline  is based on the Oxford Square Kilometre Array Simulator using a composite sky model that combines radio
foregrounds from The Galactic and Extragalactic All-Sky MWA Survey, Haslam 408MHz, and a simulated 1.5Gpc 21-cm
brightness temperature cube generated via a semi-numerical code.

@author:
    Oscar Sage David O'Hara
@email:
    osdo2@cam.ac.uk
"""

import os
import shutil
from datetime import datetime

import h5py
import numpy as np
import pandas as pd

from SKA_Power_Spectrum_and_EoR_Window.End2End.coaxial_transmission import compute_interferometer_s21
from SKA_Power_Spectrum_and_EoR_Window.End2End.OSKAR_interferometer import run_oskar
from SKA_Power_Spectrum_and_EoR_Window.End2End.generate_EoR import plot_eor
from SKA_Power_Spectrum_and_EoR_Window.End2End.logger import init_logger


def to_hdf5(gains, frequencies, folder):
    """
    Write gain model data to an HDF5 file.

    Parameters:
    -----------
    gains: numpy.ndarray
        Array of antenna gain values for each frequency.
    frequencies: numpy.ndarray
        Array of frequencies in Hz corresponding to each gain value.
    folder: str
        Path to the directory where the HDF5 file will be created.

    Returns:
    --------
    None

    Notes:
    ------
    The function creates an HDF5 file named "gain_model.h5" in the specified folder and writes two datasets
    to it: "freq (Hz)" which contains the frequency values in Hz (scaled by 1e9) and "gain_xpol" which contains
    the gain values for each frequency.

    Example:
    --------
    import numpy as np
    gains = np.array([1.0, 2.0, 3.0])
    frequencies = np.array([1e9, 2e9, 3e9])
    folder = "/path/to/directory"
    to_hdf5(gains, frequencies, folder)
    """
    with h5py.File(folder + "/gain_model.h5", "w") as hdf_file:
        hdf_file.create_dataset("freq (Hz)", data=frequencies)
        hdf_file.create_dataset("gain_xpol", data=gains)


def OSKAR_pipeline_run(max_freq=0.165,
                       min_freq=0.145,
                       channel_bandwidth=0.000125,
                       channels=160,
                       ra0_deg=60.0,
                       dec0_deg=-30.0,
                       fov=90,

                       stations='antenna_pos_core/',

                       observation_start_time_utc='2000-01-01 00:00:00',
                       observation_length_sec=0.0,
                       observation_num_time_steps=1,

                       intended_length=8.0,
                       length_variation=0.00,
                       base_temperature=298.15,
                       temp_variation=0.0,
                       cable_reflections=False,
                       loss=True,
                       z_ref=55,
                       s_s=None,
                       s_l=None,

                       eor=True,
                       dc_path='135-165MHz',

                       foregrounds=[],  # Available options include ['gleam', 'gsm', 'haslam']
                       gaussian_shape=False,

                       delete_vis=False,
                       oskar_binary=True
                       ):
    """
    Run an OSKAR pipeline simulation.

    Parameters:
    -----------
    max_freq: float, optional
        Maximum frequency to use for the simulation in GHz (default is 0.1001).
    min_freq: float, optional
        Minimum frequency to use for the simulation in GHz (default is 0.070).
    channel_bandwidth: float, optional
        Bandwidth of each channel in GHz (default is 0.000012).
    channels: int, optional
        Number of channels to use in the simulation (default is 2509).
    intended_length: float, optional
        Length of the cable used for the simulation in meters (default is 10.0).
    length_variation: float, optional
        Tolerance for the length of the cable used for the simulation as a percentage (default is 0.00).
    base_temperature: float, optional
        Temperature at which attenuation due to the thermal modification of the skin effect should be computed in
        Kelvin (default is 298.15).
    cable_reflections: bool, optional
        Flag indicating whether cable reflections should be considered (default is False).
    z_l: int, optional
        Load impedance of the cables used for the simulation in ohms (default is 60).
    dc_path: str, optional
        Path to directory containing the co-moving Mpc in local observable coordinates (default is '70-100MHz').
    eor: bool, optional
        Flag indicating whether to include the 21cm cosmological signal in the simulation (default is False).
    foregrounds: bool, optional
        Flag indicating whether to include foregrounds in the simulation (default is True).
    delete_vis: bool, optional
        Flag indicating whether to delete the visibilities after the simulation is run (default is False).

    Returns:
    --------
    None
    """

    # Get datetime of simulation for file indexing.
    date = datetime.now().strftime('%Y%m%d_%H%M%S%f')

    # Initialise the logger and insert relevant information including keyword arguments.
    logger = init_logger(date)
    logger.info(f'Client initialised with parameters: \n Max Frequency: {max_freq} \n Min Frequency: {min_freq} \n '
                f'Channel Bandwidth: {channel_bandwidth} \n Channels: {channels} \n Cable Length: {intended_length}'
                f' \n Length Tolerance: {length_variation} \n '
                f'Base Temperature: {base_temperature} \n '
                f'Temperature Max/Min Variation: {temp_variation} \n '
                f'Cable Reflections: {cable_reflections} \n '
                f'Ref Impedance: {z_ref} \n '
                f'Path to directory containing the co-moving Mpc in local observable coordinates: {dc_path} \n'
                f'Station Layout Directory: {stations} \n '
                f'21cm Cosmological Signal: {eor} \n'
                f'Foreground Signal: {foregrounds} \n'
                f'Deleting Visibilities: {delete_vis} \n'
                )

    # Copy ./antenna_pos in order to generate a new telescope model to which gain_models may be applied.
    try:
        logger.info(f'Creating new telescope model from the antenna positions.')

        # Create temp dir copy of antenna pos + gains
        telescope_dir = date + '_telescope_model/'
        shutil.copytree('SKA_Power_Spectrum_and_EoR_Window/End2End/' + stations, telescope_dir)

        logger.info(f'Success, the telescope model was created: {telescope_dir}.\n')
    except Exception:
        logger.exception('The following exception was raised: \n ')
        raise

    if cable_reflections is True:
        # Generate S21 parameters/gain_models for each antenna of the array.
        try:
            logger.info(f'Generating antenna S21 scattering parameters given the initial given arguments.')
            antenna_info = compute_interferometer_s21(max_freq=max_freq,
                                                      min_freq=min_freq,
                                                      channels=channels,
                                                      channel_bandwidth=channel_bandwidth,
                                                      loss=loss,
                                                      intended_length=intended_length,
                                                      length_variation=length_variation,
                                                      base_temperature=base_temperature,
                                                      temp_variation=temp_variation,
                                                      z_ref=z_ref,
                                                      s_s=s_s,
                                                      s_l=s_l,
                                                      stations=stations)
            logger.info(f'Success, the S21 Scattering parameters have been generated.')
        except Exception:
            logger.exception('The following exception was raised: \n')
            raise

        # Save the S-parameters for each element corresponding to the station map described by the telescope model.
        try:
            logger.info(f'Saving the S21 parameters for each station to the telescope model as gain_model.h5 files.')
            for n in range(len(os.listdir(telescope_dir)) - 2):
                rows = antenna_info[pd.DataFrame(antenna_info.station.tolist()).isin([n]).values]['phasor'].values
                data = []
                [data.append(np.array(rows[i])) for i in range(len(rows))]
                to_hdf5(list(np.array([np.transpose(data)])),
                        np.arange(min_freq * 1e9, max_freq * 1e9, channel_bandwidth * 1e9),
                        date + '_telescope_model/station' + str(n).rjust(3, '0'))
            logger.info('Success, telescope model now includes antenna gain models in the form of S21 parameters. \n')
        except Exception:
            logger.exception('The following exception was raised: \n')
            raise

    # Run OSKAR using the given telescope model.
    try:
        logger.info('Running OSKAR interferometer simulations with the given telescope model.')

        # Run OSKAR for the generated telescope model.
        run_oskar(date, min_freq, channels, channel_bandwidth, ra0_deg, dec0_deg, fov, observation_start_time_utc,
                  observation_length_sec, observation_num_time_steps, eor, foregrounds, gaussian_shape, dc_path,
                  oskar_binary)

        logger.info(f'Success, OSKAR visibilities have been generated to {date}_vis. \n')
    except Exception:
        logger.exception('The following exception was raised: \n')
        raise

    # Delete telescope model (approx 5 GB) to free storage space occupied by redundant data.
    try:
        logger.info(f'Deleting used telescope model: {telescope_dir}.')

        # Removing telescope model to conserve storage space.
        shutil.rmtree(telescope_dir)

        logger.info('Success. \n')
    except Exception:
        logger.exception('The following exception was raised: \n')
        raise

    # Delete sky map folder for clean-up.
    try:
        logger.info(f'Deleting used sky model: {date}_sky_maps/.')

        # Removing telescope model to conserve storage space.
        shutil.rmtree(date + '_sky_maps/')

        logger.info('Success. \n')
    except Exception:
        logger.exception('The following exception was raised: \n')
        raise

    # Plotting the EoR window plots using the control.ms and the newly generated visibilities.
    if oskar_binary is True:
        try:
            logger.info(f'Plotting the EoR windows for visibilities {date}_vis')

            # Create directory to write SKA window plots to.
            result_dir = date + '_results'
            os.mkdir(result_dir)

            # Plotting data generated by OSKAR in order to create the EoR windows.
            plot_eor(date + '_vis', result_dir, min_freq, max_freq, channels, channel_bandwidth,
                     observation_num_time_steps)
            logger.info('Success. \n')
        except Exception:
            logger.exception('The following exception was raised: \n')
            raise

    # Delete the OSKAR visibilities (approx 40 GB) to free storage space occupied by redundant data.
    if delete_vis is True:
        try:
            logger.info(f'Deleting used visibilities {date}_vis')

            # Removing visibilities to conserve storage space.
            shutil.rmtree(date + '_vis')

            logger.info('Success. \n')
        except Exception:
            logger.exception('The following exception was raised: \n')
            raise


if __name__ == '__main__':
    OSKAR_pipeline_run()
