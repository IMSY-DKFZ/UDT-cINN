import numpy as np


class ImagingSystem(object):
    r"""
    Encapsulates all the information needed to link spectrometer/reflectance
    measurements to camera measurements which are dependent on the cameras
    imaging system (the cam_sensitivity used, quantum efficiency, ...)

    More concretely the following equation is assumed to hold:

    .. math::
        C_{j,k} & \propto \frac{\int_{l=0}^m q_l f_{k,l} i_l r_{j,l}}
                               {\int_{l=0}^m q_l f_{k,l} i_l} \\
        & =   \frac{\int_{l=0}^m  q_l f_{k,l} i_l \frac{s_{j,l} - d_l}{w_l - d_l}}
                    {\int_{l=0}^m q_l f_{k,l} i_l}\\
        & =   \frac{\int_{l=0}^m  q_l f_{k,l} i_l \frac{s_{j,l} - d_l}{i_l}}
                    {\int_{l=0}^m q_l f_{k,l} i_l}\\
        & =   \frac{\int_{l=0}^m  q_l f_{k,l} (s_{j,l} - d_l)}
                    {\int_{l=0}^m q_l f_{k,l} i_l}\\
        & =   \frac{\int_{l=0}^m  q_l f_{k,l} (s_{j,l} - d_l)}
                    {\int_{l=0}^m q_l f_{k,l} (w_{l} - d_l)}

    .. list-table:: Dimensions
        :widths: 25 25 50
        :header-rows: 1

        * - dimension
          - rolling index
          - description
        * - n
          - j
          - number of measurements
        * - v
          - k
          -  number of wavelengths measured by imaging system
        * - m
          - l
          - number of wavelengths measured by spectrometer

    .. list-table:: Parameters
        :widths: 25 25 50
        :header-rows: 1

        * - parameters
          - dimensions
          - description
        * - wavelengths
          - m
          - wavelengths measured by the spectrometer
        * - cam_sensitivity
          - v * m
          - initial estimate on the filter sensitivities
        * - q
          - m
          - initial estimate on the quantum efficiency of the camera. For each wavelength measured by the spectrometer gives a value on how big the cameras quantum efficiency is.
        * - w
          - m
          - *white* measurements, contains the loaded irradiance from optical components passed to :code:`self.__init_()`
        * - d
          - m
          - *dark* measurements
        * - s
          - n * m
          - spectrometer measurements
        * - c
          - n * v
          - measurements made by camera
    """
    def __init__(self, wavelengths, cam_sensitivity, q=None, w=None, d=None):
        """

        :param wavelengths:
        :param cam_sensitivity:
        :param q:
        :param w:
        :param d:
        """
        self.wavelengths = wavelengths
        m = len(self.wavelengths)
        v, m2 = cam_sensitivity.shape
        if m != m2:
            raise ValueError("number of wavelengths in the filter specification " +
                             str(m2) +
                             " does not match number of wavelengths " + str(m))
        if q is None:
            q = np.ones(m)
        if w is None:
            w = np.ones(m)
        if d is None:
            d = np.zeros(m)

        self.q = np.squeeze(q)
        self.F = cam_sensitivity
        self.w = np.squeeze(w)
        self.d = np.squeeze(d)

    def get_v(self):
        """
        get the number of bands measured by the imaging system

        :return:
        """
        return self.F.shape[0]

    def get_nr_bands(self):
        """
        Alias to :code:`self.get_v()`

        :return: number of bands of the imaging system (v)
        """
        return self.get_v()

    def __repr__(self):
        rep_str = 'Imaging system:\n' \
                  'Filters:\n' + \
                  str(self.F) + '\n' + \
                  'White: \n' + str(self.w) + \
                  'Quantum efficiency: \n' + str(self.q) + \
                  'Parsing wavelenghts: \n' + str(self.wavelengths)
        return rep_str


def spectral_reflectance_to_camera_color(imaging_system, r):
    """
    Transforms spectral reflectances into simulated camera intensities (camera color)

    :param imaging_system: imaging system created with class :class:`susi.mc.camera.ImagingSystem`
    :param r: array containing reflectances with dimensions (nr_samples x nr_bands)
    :return: array containing simulated camera intensities
    """
    i = imaging_system  # short alias for imaging system
    # initialize array on which to save values
    camera_color = np.zeros((_nr_samples_to_transform(r), i.get_nr_bands()))

    # iterate over bands of imaging system
    for k in range(i.get_v()):
        combined_imaging_system = i.q * i.F[k, :] * (i.w - i.d)
        vectorized_response = combined_imaging_system * r
        # integrate over wavelengths to get camera response for band k:
        color_k = np.trapz(vectorized_response, i.wavelengths)
        camera_color[:, k] = color_k
    return np.squeeze(camera_color)


def spectral_irradiance_to_camera_color(imaging_system, s):
    """
    Transforms spectral irradiance into simulated camera intensities (camera color)

    :param imaging_system: imaging system created with class :class:`susi.mc.camera.ImagingSystem`
    :param s: array containing irradiance with dimensions (nr_samples x nr_bands)
    :return: array containing simulated camera intensities
    """
    i = imaging_system  # short alias for imaging system
    camera_color = np.zeros((_nr_samples_to_transform(s), i.get_nr_bands()))

    # iterate over bands of imaging system
    for k in range(i.get_nr_bands()):
        combined_imaging_system = i.q * i.F[k, :]
        vectorized_response = combined_imaging_system * (s - i.d)
        # camera response for band k:
        color_k = np.trapz(vectorized_response, i.wavelengths)
        camera_color[:, k] = color_k
    return np.squeeze(camera_color)


def camera_color_to_camera_reflectance(imaging_system, camera_color):
    """
    Transforms camera intensities to camera reflectances

    :param imaging_system: imaging system created with class :class:`susi.mc.camera.ImagingSystem`
    :param camera_color: array containing simulated camera intensities with dimensions (nr_samples x nr_bands)
    :return: array containing simulated camera reflectances
    """
    if len(camera_color.shape) == 1:
        camera_color = camera_color[np.newaxis, :]
    # normalize each band
    white = get_white_color(imaging_system)
    camera_reflectance = camera_color / white[np.newaxis, ...]
    return np.squeeze(camera_reflectance)


def get_white_color(imaging_system: ImagingSystem):
    """
    computes "white" reference for normalization of camera intensities, depends only on imaging system

    :param imaging_system: imaging system created with class :class:`susi.mc.camera.ImagingSystem`
    :return: array containing "white" reference
    """
    i = imaging_system  # short alias for imaging system

    white = np.zeros(i.get_nr_bands())
    for k in range(i.get_nr_bands()):
        combined_imaging_system = i.q * i.F[k, :] * (i.w - i.d)
        white[k] = np.trapz(combined_imaging_system, i.wavelengths)
    return white


def transform_reflectance(imaging_system, r):
    """
    Given a set of reflectances (nxm), transform them to what camera
    reflectance space (no noise added).

    :param imaging_system: imaging system created with class :class:`susi.mc.camera.ImagingSystem`
    :param r: set of reflectance measurements (nxm). These can e.g. be the output of a Monte Carlo simulation.
    :return:the measurement transformed to show what the imaging system would measure (nxv)
    """
    camera_color = spectral_reflectance_to_camera_color(imaging_system, r)
    camera_reflectance = camera_color_to_camera_reflectance(imaging_system, camera_color)
    return camera_reflectance


def transform_color(imaging_system, s):
    """
    Given a set of spectrometer irradiance measurements (nxm),
    transform them to what the camera reflectance space (no noise added).
    The difference to transform_reflectance is that S is not dark and white
    light corrected.

    :param imaging_system: imaging system created with class :class:`susi.mc.camera.ImagingSystem`
    :param s: set of spectrometer measurements (nxm)
    :return: the measurement transformed to show what the imaging system would measure (nxv)
    """
    camera_color = spectral_irradiance_to_camera_color(imaging_system, s)
    camera_reflectance = camera_color_to_camera_reflectance(imaging_system, camera_color)
    return np.squeeze(camera_reflectance)


def _nr_samples_to_transform(x):
    """
    Computes the number of samples to be transformed. Uses dimension 0 of array as # samples

    :param x: array containing samples, dimension 0 corresponds to # samples
    :return: int, number of samples in array
    """
    if len(x.shape) == 1:
        n = 1
    else:
        n = x.shape[0]
    return n
