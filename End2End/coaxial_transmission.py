"""
This script implements a parametric model to account for the effects of a coaxial transmission line and on the
scattering parameters in terms of the cables bulk material parameters. This code is consistent with numerical simulation
ran in CST cable studio to within 5%.

@author:
    Oscar Sage David O'Hara
@email:
    osdo2@cam.ac.uk
"""


import noise
import numpy as np
import pandas as pd
import scipy.constants as const
import skrf as rf
from skrf import Network, Frequency


class cable_decay:
    """
    The electromagnetic properties of a Coaxial transmission line may be analytically computed using the second order closed
    form differential Telegrapher's equations given the cable structure and composite material bulk parameters.

    This Class calculates these properties and S21 parameters as a function of length & temperture.
    """

    def __init__(self, max_freq, min_freq, channels, channel_bandwidth, a, b, c, rho_in, rho_out, mu_in, mu_out,
                 roughness, eps_dielectric, rho_dielectric, mu_dielectric, tcr_in, tcr_out, z_ref, s_s, s_l):
        """
        Initializes the cable_decay class.

        Parameters:
            - max_freq (float): Maximum frequency (GHz) for which the electrical properties should be calculated.
            - min_freq (float): Minimum frequency (GHz) for which the electrical properties should be calculated.
            - channels (int): Number of channels in the bandwidth.
            - channel_bandwidth (float): Bandwidth (GHz) per channel.
            - a (float): Inner conductor radius (mm).
            - b (float): Outer conductor radius (mm).
            - c (float): Dielectric radius (mm).
            - rho_in (float): Inner conductor resistivity (Ohm.m).
            - rho_out (float): Outer conductor resistivity (Ohm.m).
            - mu_in (float): Inner conductor permeability (H/m).
            - mu_out (float): Outer conductor permeability (H/m).
            - roughness (float): Inner conductor surface roughness (microns).
            - eps_dielectric (float): Dielectric permittivity (F/m).
            - rho_dielectric (float): Dielectric resistivity (Ohm.m).
            - mu_dielectric (float): Dielectric permeability (H/m).
            - tcr_in (float): Temperature coefficient of resistance for the inner conductor.
            - tcr_out (float): Temperature coefficient of resistance for the outer conductor.
            - z_ref (complex): Refrence impedance (Ohm).
        """

        self.max_freq = max_freq * 10 ** 9  # Convert GHz to Hz.
        self.min_freq = min_freq * 10 ** 9  # Convert GHz to Hz.
        self.channels = channels
        self.channel_bandwidth = channel_bandwidth * 10 ** 9
        self.a = a
        self.b = b
        self.c = c
        self.rho_in = rho_in
        self.rho_out = rho_out
        self.mu_in = mu_in * const.mu_0  # Convert to core permeability.
        self.mu_out = mu_out * const.mu_0  # Convert to shield permeability.
        self.roughness = roughness
        self.eps_dielectric = eps_dielectric * const.epsilon_0
        self.rho_dielectric = rho_dielectric
        self.mu_dielectric = mu_dielectric * const.mu_0

        # Tempeture Coeficient of Resistance @ 20 degrees Celcius:
        # https://www.allaboutcircuits.com/textbook/direct-current/chpt-12/temperature-coefficient-resistance/
        self.tcr_in = tcr_in
        self.tcr_out = tcr_out

        self.z_ref = z_ref
        self.s_s = s_s
        self.s_l = s_l

        # Create all frequency & wavelengths for calculations.
        self.frequencies = np.arange(self.min_freq, self.max_freq, self.channel_bandwidth)  # Hz.
        self.wavelengths = const.speed_of_light / self.frequencies  # meters.
        self.angular_frequencies = 2 * np.pi * self.frequencies

        self.get_capacitance()
        self.get_inductance()
        self.get_conductance()

    def get_capacitance(self):
        """
        Calculate the capacitance of a cylindrical capacitor based on the given parameters.

        Returns:
        capacitance (float): The capacitance of the cylindrical capacitor.

        Parameters:
        self (object): The object representing the cylindrical capacitor.
        self.eps_dielectric (float): The dielectric constant of the material between the two cylinders.
        self.a (float): The radius of the inner cylinder.
        self.b (float): The radius of the outer cylinder.
        """
        self.capacitance = (2 * np.pi * self.eps_dielectric) / np.log(self.b / self.a)
        return self.capacitance

    def get_inductance(self):
        """
        Calculate the inductance of a cylindrical inductor based on the given parameters.

        Returns:
        inductance (float): The inductance of the cylindrical inductor.

        Parameters:
        self (object): The object representing the cylindrical inductor.
        self.mu_dielectric (float): The permeability of the material inside the cylinder.
        self.a (float): The radius of the inner cylinder.
        self.b (float): The radius of the outer cylinder.
        """
        self.inductance = (self.mu_dielectric) / (2 * np.pi) * np.log(self.b / self.a)
        return self.inductance

    def get_conductance(self):
        """
        Calculate the conductance of a cylindrical conductor based on the given parameters.

        Returns:
        conductance (float): The conductance of the cylindrical conductor.

        Parameters:
        self (object): The object representing the cylindrical conductor.
        self.rho_dielectric (float): The resistivity of the material inside the cylinder.
        self.a (float): The radius of the inner cylinder.
        self.b (float): The radius of the outer cylinder.
        """
        self.conductance = (2 * np.pi / self.rho_dielectric) / np.log(self.b / self.a)
        return self.conductance

    def get_resistance(self, temp):
        """
        Calculate the resistance of a cylindrical transmission line based on the given parameters and temperature.

        Returns:
        resistance (float): The resistance of the cylindrical transmission line.

        Parameters:
        self (object): The object representing the cylindrical transmission line.
        temp (float): The temperature of the transmission line in Kelvin.
        self.rho_in (float): The resistivity of the inner conductor.
        self.rho_out (float): The resistivity of the outer conductor.
        self.tcr_in (float): The temperature coefficient of resistance for the inner conductor.
        self.tcr_out (float): The temperature coefficient of resistance for the outer conductor.
        self.a (float): The radius of the inner conductor.
        self.b (float): The radius of the outer conductor.
        self.angular_frequencies (float): The angular frequency of the transmission line in radians per second.
        self.mu_in (float): The permeability of the material inside the inner conductor.
        self.mu_out (float): The permeability of the material between the inner and outer conductors.
        """
        delta_in = np.sqrt(2 * self.rho_in / (self.angular_frequencies * self.mu_in))
        delta_out = np.sqrt(2 * self.rho_out / (self.angular_frequencies * self.mu_out))

        self.resistance = 1 / (2 * np.pi) * (self.rho_in * (1 + self.tcr_in * (temp - 293.15)) / (delta_in * self.a) +
                                             self.rho_out * (1 + self.tcr_out * (temp - 293.15)) / (delta_out * self.b))
        return self.resistance

    def propagation_const(self):
        """
        Calculate the propagation constant, phase constant, and attenuation constant of a cylindrical transmission line.

        Returns:
        phase_const (float): The phase constant of the transmission line.
        attenuation (float): The attenuation constant of the transmission line.

        Parameters:
        self (object): The object representing the cylindrical transmission line.
        """
        self.gamma = np.sqrt((self.resistance + 1j * self.angular_frequencies * self.inductance) * (
                self.conductance + 1j * self.angular_frequencies * self.capacitance))
        self.phase_const = self.gamma.imag
        self.attenuation = self.gamma.real
        return self.phase_const

    def coax_impedence(self):
        """
        Calculate the characteristic impedance of a coaxial transmission line.

        Returns:
        z_0 (float): The characteristic impedance of the coaxial transmission line.

        Parameters:
        self (object): The object representing the coaxial transmission line.
        self.resistance (float): The resistance of the transmission line.
        self.inductance (float): The inductance of the transmission line.
        self.conductance (float): The conductance of the transmission line.
        self.capacitance (float): The capacitance of the transmission line.
        self.angular_frequencies (float): The angular frequency of the transmission line in radians per second.
        """
        self.z_0 = np.sqrt((self.resistance + 1j * self.angular_frequencies * self.inductance) / (
                self.conductance + 1j * self.angular_frequencies * self.capacitance))
        return self.z_0

    def get_S21(self, length, temp):
        """
        Calculates the S21 response for a given coaxial cable length, temperature, number of reflections, and attenuation value.

        Args:
        length (float): The length of the cable in meters.
        temp (float, optional): The temperature in Kelvin. Defaults to 293.15.
        n (int, optional): The number of reflections to consider. Defaults to 10.
        atten_bead (float, optional): The attenuation value in dB. Defaults to 0.0.
        cable_reflections (bool, optional): Whether or not to include reflections in the cable. Defaults to True.

        Returns:
        complex: The S21 response.
        """

        self.get_resistance(temp)
        self.coax_impedence()
        self.propagation_const()

        S_11 = np.divide((self.z_0 ** 2 - self.z_ref ** 2) * np.sinh(self.gamma * length),
                         (self.z_0 ** 2 + self.z_ref ** 2) * np.sinh(
                             self.gamma * length) + 2 * self.z_ref * self.z_0 * np.cosh(self.gamma * length))
        S_22 = S_11

        S_12 = np.divide(2 * self.z_ref * self.z_0,
                         (self.z_0 ** 2 + self.z_ref ** 2) * np.sinh(
                             self.gamma * length) + 2 * self.z_ref * self.z_0 * np.cosh(self.gamma * length))
        S_21 = S_12
        return S_11, S_12, S_21, S_22

    def get_gain(self, length, temp=293.15):
        """
        Calculates the voltage at the end of a coaxial cable of a given length with an optional attenuation value.

        Args:
        length (float): The length of the cable in meters.
        n (int, optional): The number of reflections to consider. Defaults to 3.
        atten_bead (float, optional): The attenuation value in dB. Defaults to 0.0.

        Returns:
        complex: The voltage at the end of the cable.
        """

        S_11, S_12, S_21, S_22 = self.get_S21(length=length, temp=temp)

        if self.s_s and self.s_l is not None:
            s = np.transpose(np.array([[S_11, S_12], [S_21, S_22]]), (2, 0, 1))

            freq = Frequency.from_f(self.frequencies, 'mhz')

            ntwk_s = Network(frequency=freq, s=self.s_s, z0=self.z_ref, name='Source 2-port')
            ntwk_0 = Network(frequency=freq, s=s, z0=self.z_ref, name='Coaxial Transmission Line 2-port')
            ntwk_l = Network(frequency=freq, s=self.s_l, z0=self.z_ref, name='Load 2-port')

            ntwk = rf.cascade_list([ntwk_s, ntwk_0, ntwk_l])

            # Complex Linear Gain
            gain = np.transpose(ntwk.s, (1, 2, 0))[1, 0, :]
        else:
            gain = np.abs(S_21)
        return gain


def equirectangular_approx(lat1, lon1, lat2, lon2):
    """Converts latitude and longitude to polar coordinates using an equirectangular approximation.

    Parameters
    ----------
        lat_rel : float
            Latitude relative to the reference latitude.
        lon_rel : float
            Longitude relative to the reference longitude.
        ref_lat : float
            Reference latitude in degrees.
        ref_lon : float
            Reference longitude in degrees.

    Returns
    -------
        rho : float
            Distance from the reference point in meters.
        phi : float
            Angle with respect to the reference point in radians.
    """
    r = 6371 * 10 ** 3  # radius of Earth in Meters

    theta_1 = lat1 * np.pi / 180
    theta_2 = lat2 * np.pi / 180
    delta_theta = (lat2 - lat1) * np.pi / 180
    delta_lam = (lon2 - lon1) * np.pi / 180

    x = np.sin(delta_lam) * np.cos(theta_2)
    y = np.cos(theta_1) * np.sin(theta_2) - (np.sin(theta_1) * np.cos(theta_2) * np.cos(delta_lam))
    phi = (np.arctan2(x, y) + 3 * np.pi) % (2 * np.pi)

    a = np.sin(delta_theta / 2) ** 2 + np.cos(theta_1) * np.cos(theta_2) * np.sin(delta_lam / 2) ** 2
    c = 2 * np.arctan(np.sqrt(a) / np.sqrt(1 - a))
    rho = r * c

    return [rho, phi]  # meters, degrees


def polar_to_cart(rho, phi):  # meters, degrees
    """Converts polar coordinates to Cartesian coordinates.

    Parameters
    ----------
        rho : float
            Distance from the origin in meters.
        phi : float
            Angle with respect to the x-axis in radians.

    Returns
    -------
        list: Cartesian coordinates [x, y].
    """
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return [x, y]


def get_antenna_pos(stations):
    """Returns the positions of the SKA antennas in the reference frame of the SKA core.

    Returns
    -------
        pandas.DataFrame: DataFrame containing the station index and the Cartesian coordinates [x, y] of each antenna.
    """
    centre = [116.7644482, -26.82472208]  # lat , lon
    station_pos = pd.read_csv('SKA_Power_Spectrum_and_EoR_Window/End2End/' + stations + 'layout_wgs84.txt',
                              header=None, names=["latitude", "longitude"])
    station_pos['lat_rel'] = (station_pos['latitude'] - centre[0])
    station_pos['lon_rel'] = (station_pos['longitude'] - centre[1])
    station_pos['rho'], station_pos['phi'] = equirectangular_approx(station_pos['lat_rel'], station_pos['lon_rel'], 0.0,
                                                                    0.0)
    station_pos['x'], station_pos['y'] = polar_to_cart(station_pos['rho'], station_pos['phi'])

    antenna_info = pd.DataFrame(columns=['station', 'x', 'y'])
    for i, x, y in zip(range(len(station_pos['lat_rel'])), station_pos['x'], station_pos['y']):
        df = pd.read_csv('SKA_Power_Spectrum_and_EoR_Window/End2End/' + stations + 'station' +
                         str(i).rjust(3, '0') + '/layout.txt', header=None,
                         names=["delta_x", "delta_y"])
        df['delta_x'], df['delta_y'] = df['delta_x'] + x, df['delta_y'] + y
        df['station'] = i
        df = df[['station', "delta_x", "delta_y"]]
        antenna_info = antenna_info.append(pd.DataFrame(df.to_dict('split')['data'], columns=['station', "x", "y"]),
                                           ignore_index=True)
    return antenna_info


def perlin_noise_map(shape=(1000, 1000), scale=np.random.uniform(800.0, 1200.0), octaves=np.random.randint(10, 20),
                     persistence=0.5, lacunarity=2.0):
    """
    Generates a 2D Perlin noise map with the specified parameters.

    Parameters:
    -----------
    shape : tuple of int, optional
        The dimensions of the output map as a tuple of (height, width). Default is (1000, 1000).
    scale : float, optional
        The scaling factor of the noise function. Higher values result in more fine-grained noise. Default is a random float between 800.0 and 1200.0.
    octaves : int, optional
        The number of octaves (iterations) used in the noise generation. Higher values result in more detailed noise. Default is a random integer between 10 and 20.
    persistence : float, optional
        The persistence of the noise function. Higher values result in more contrast between high and low values. Default is 0.5.
    lacunarity : float, optional
        The lacunarity of the noise function. This parameter controls the increase in frequency between octaves. Default is 2.0.

    Returns:
    --------
    world : ndarray of float
        A 2D numpy array of shape (height, width) containing the generated Perlin noise map.
    """
    world = np.zeros(shape)
    for i in range(shape[0]):
        for j in range(shape[1]):
            world[i][j] = noise.snoise2(i / scale,
                                        j / scale,
                                        octaves=octaves,
                                        persistence=persistence,
                                        lacunarity=lacunarity,
                                        base=0)
    return world


def compute_interferometer_s21(max_freq, min_freq, channels, channel_bandwidth, intended_length, length_variation,
                               base_temperature, temp_variation, z_ref, s_s, s_l, stations):
    """
    Compute the S21 signal for a radio interferometer system.

    Parameters:
    -----------
    max_freq : float
        The maximum frequency (in MHz) for the S21 signal.
    min_freq : float
        The minimum frequency (in MHz) for the S21 signal.
    channels : int
        The number of channels used to sample the S21 signal.
    channel_bandwidth : float
        The bandwidth (in MHz) of each channel.
    intended_length : float
        The intended length (in meters) of the coaxial cable.
    length_variation : float
        The relative variation of the cable length, expressed as a standard deviation.
    base_temperature : float
        The base temperature (in Kelvin) for the S21 signal calculation.
    cable_reflections : bool
        Whether to include reflections from the coaxial cable in the S21 signal calculation.
    z_l : complex
        The load impedance of the S21 signal.
    z_s : complex
        The generator impedance of the S21 signal.

    Returns:
    --------
    antenna_info : DataFrame
        A pandas DataFrame containing information about the antennas in the system, including their positions,
        cable lengths, delta temperatures, and S21 phasors.
    """
    antenna_info = get_antenna_pos(stations)

    world = perlin_noise_map() * temp_variation

    # Initalise an RG58 Coaxial Cable across the 1480 channels spaning freq bandwidth [72.0, 108.975]MHz.
    ska = cable_decay(max_freq=max_freq, min_freq=min_freq, channels=channels, channel_bandwidth=channel_bandwidth,
                      a=0.0004572, b=0.0014732, c=0.0017272, rho_in=1.71e-8, rho_out=1.71e-8,
                      mu_in=1.0, mu_out=1.0, roughness=0.0, eps_dielectric=2.12, rho_dielectric=1e18,
                      mu_dielectric=1, tcr_in=0.00404, tcr_out=0.00404, z_ref=z_ref, s_s=s_s, s_l=s_l)

    antenna_info['cable_length'] = np.random.normal(intended_length, intended_length * length_variation,
                                                    len(antenna_info.index))

    shape = (1000, 1000)
    antenna_info['delta_t'] = [base_temperature + world[round((x / 40000) * shape[0] / 2 + shape[0] / 2)][
        round((y / 40000) * shape[1] / 2 + shape[1] / 2)] for x, y in zip(antenna_info['x'], antenna_info['y'])]

    antenna_info['phasor'] = [ska.get_gain(l, dt) for l, dt in
                              zip(antenna_info['cable_length'], antenna_info['delta_t'])]
    return antenna_info
