import os
from functools import reduce
from typing import Iterable, List, Union

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from src.utils.camera import ImagingSystem, transform_reflectance


class Msi:
    """ a multi spectral image stack consisting of:

    image:      a rows x columns x nrWavelengths dimensional numpy array
    properties: additional, application dependent properties (dictionary)

    If no properties are given, but an image is passed the mandatory "wavelengths"
    property will be added automatically, wavelengths will be set to
    0,1,..nrWavelengths
    """

    def __init__(self, image=None, properties=None):
        if image is None:
            image = np.array([])
        if properties is None:
            properties = {}
        self._image = image
        self._properties = properties
        self._assure_basic_properties()

        self._test_image()

    def get_image(self):
        return self._image

    def set_image(self, image, wavelengths=None):
        """
        Put a new image into this msi
        Args:
            image: the rows x columns x nrWavelengths dimensional array
                   np.array.
            wavelengths: a np.array of size nrWavelengths. If the number of
                         wavelengths hasn't change this is not needed.
        """
        self._image = image
        if wavelengths is not None:
            self.set_wavelengths(wavelengths)
        self._assure_basic_properties()
        self._test_image()

    def get_wavelengths(self):
        """ shortcut to get the wavelengths property
        The wavelengths are given in [m] units and need not be sorted. """
        if 'wavelengths' not in self.get_properties():
            return None
        return self._properties['wavelengths']

    def set_wavelengths(self, wavelengths):
        """ shortcut to set the wavelengths property """
        w_prop = {"wavelengths": wavelengths}
        self.add_property(w_prop)
        self._test_image()

    def get_properties(self):
        return self._properties

    def add_property(self, new_property):
        """ add a new property(ies) to the existing properties """
        self._properties.update(new_property)
        self._test_image()

    def set_mask(self, mask):
        """" applies a masked to the Msi. After this call, the image is of
        type MaskedArray. If the image was already masked, the existing
        masked will be "or ed" with the new mask. mask is a boolean array of
        the same shape as self.get_image()

        Args:
            mask: a mask of the same size as the image. 1s stand for pixels
                  masked out, 0s for pixels not masked."""
        if not isinstance(self.get_image(), np.ma.MaskedArray):
            self.set_image(np.ma.masked_array(self.get_image(), mask,
                           fill_value=999999))
        else:
            self.get_image()[mask] = np.ma.masked

    def apply_to_image(self, function):
        """
        Apply the function to the image. E.g. the -log transformation would be
        `self.apply(lambda minuslog: - np.log)`
        :param function:
        :return: nothing, image is changed in place
        """
        self.set_image(function(self.get_image()))

    def size(self):
        return self._image.shape

    def __eq__(self, other):
        """
        overwrite the == operator
        Two Msi s are the same if they contain the same image and properties.
        Note: properties not implemented yet!
        """
        if isinstance(other, Msi):
            same = np.array_equal(other.get_image(), self.get_image())
            return same
        return NotImplemented

    def __ne__(self, other):
        """ != operator implemented by inverting to =="""
        result = self.__eq__(other)
        if result is NotImplemented:
            return result
        return not result

    def _assure_basic_properties(self):
        """
        helper method to automatically add the basic properties:
        wavelength
        to the msi if not added explicitly. basic wavelengths will just be
        integers from 0 to 1
        """
        if self._image.size > 0 and (
                ("wavelengths" not in self._properties.keys() or
                 self._properties["wavelengths"].size == 0)):
            self._properties["wavelengths"] = np.arange(self._image.shape[-1])
        if self._image.size == 0 and "wavelengths" not in self._properties.keys():
            self._properties["wavelengths"] = np.array([])

    def _test_image(self):
        """
        helper method which tests for the integrity of the msi.
        E.g. the number of wavelengths must match the number of bands.
        """
        # either both image and wavelength property are empty
        if self._image.size == 0 and len(self._properties["wavelengths"]) != 0:
            raise RuntimeError("dimension of image and wavelength mismatch: " +
                               "image size is zero, but wavelengths are set")
        # or both are same
        elif self._image.shape[-1] != len(self._properties["wavelengths"]):
            raise RuntimeError("dimension of image and wavelength mismatch: " +
                               "image size and wavelengths do not match")

    def describe(self):
        """
        Computes statistics on multispectral image. A message is issued if nan or infinite values are found in image
        data. If nan values are found, statistics are computed ignoring such values. The following parameters are
        computed: `minimum, maximum, mean. median & standard deviation`
        :return: None
        """
        if isinstance(self.get_image(), np.ndarray):
            im = self.get_image()
            n_bands = im.shape[-1]
            spatial_dims = im.shape[:1]
            if np.isnan(im).any():
                print(f"There are nan values in data here: {np.where(np.isnan(im))}")
                print("Computing statistics ignoring nan values ")
                min_v = [np.nanmin(im[..., i]) for i in np.arange(n_bands)]
                max_v = [np.nanmax(im[..., i]) for i in np.arange(n_bands)]
                mean_v = [np.nanmean(im[..., i]) for i in np.arange(n_bands)]
                median_v = [np.nanmedian(im[..., i]) for i in np.arange(n_bands)]
                std_v = [np.nanstd(im[..., i]) for i in np.arange(n_bands)]
            else:
                min_v = [im[..., i].min() for i in np.arange(n_bands)]
                max_v = [im[..., i].max() for i in np.arange(n_bands)]
                mean_v = [im[..., i].mean() for i in np.arange(n_bands)]
                median_v = [np.median(im[..., i]) for i in np.arange(n_bands)]
                std_v = [im[..., i].std() for i in np.arange(n_bands)]
            if not np.isfinite(im).any():
                print(f"There are infinite values in data here: {np.where(1 - np.isfinite(im))}")
            print(f"Minimum values for each channel are: {min_v},\n\n Maximum values for each channel are: {max_v}\n\n",
                  f"Mean values for each channel are: {mean_v},\n\n Median values for each channel are: {median_v}\n\n",
                  f"Standard deviations for each channel are: {std_v},\n\n Number of channels (bands): {n_bands}\n\n",
                  f"Spatial dimensions are: {spatial_dims}")


class SpectrometerReader:
    def __init__(self):
        super().__init__()

    def read(self, file_to_read):
        transformed = ""
        replacements = {',': '.', '\r\n': ''}
        with open(file_to_read) as infile:
            for line in infile:
                for src, target in replacements.items():
                    line = line.replace(src, target)
                transformed = "\n".join([transformed, line])

        for num, line in enumerate(transformed.splitlines(), 1):
            if ">>>>>Begin" in line:
                break

        for num_end, line in enumerate(transformed.splitlines(), 1):
            if ">>>>>End" in line:
                num_end -= 1
                break
        string_only_spectrum = "\n".join(transformed.splitlines()[num: num_end])
        data_vector = np.fromstring(string_only_spectrum,
                                    sep="\t").reshape(-1, 2)
        msi = Msi(data_vector[:, 1],
                  {'wavelengths': data_vector[:, 0] * 10 ** -9})
        return msi


def get_spectrometer_measurement(filename: str) -> np.ndarray:
    """
    Returns np.ndarray containing the image
    :param filename: str, file to read
    :return: np.ndarray
    """
    spectro_reader = SpectrometerReader()
    return spectro_reader.read(filename).get_image()


def get_spectrometer_wavelengths(filename: str) -> np.ndarray:
    """
    Returns np.ndarray containing the wavelengths
    :param filename: str, file to read
    :return: np.ndarray
    """
    spectro_reader = SpectrometerReader()
    return spectro_reader.read(filename).get_wavelengths()


def get_spectrometer_measurement_series(filename: str) -> pd.Series:
    """
    Uses get_spectrometer_measurement and get_spectrometer_wavelengths and puts them together in one pd.Series
    :param filename: str, file to read
    :return: pd.Series
    """
    meas = get_spectrometer_measurement(filename)
    w = get_spectrometer_wavelengths(filename)
    return pd.Series(data=np.squeeze(meas), index=w)


def to_wav_df(df: pd.DataFrame, wav) -> pd.DataFrame:
    data_new = to_wav(np.array(df.columns).astype(float), np.array(df.values), np.array(wav))
    return pd.DataFrame(data=data_new, columns=wav, index=df.index)


def to_wav_series(series: pd.Series, wav) -> pd.Series:
    data_new = to_wav(np.array(series.index).astype(float), np.array(series.values), np.array(wav))
    return pd.Series(data=np.squeeze(data_new), index=wav, dtype=float)


def to_wav(old_wav: np.ndarray, data: np.ndarray, f_wav: np.ndarray):
    """
    interpolate the spectrometer values to fit the wavelengths recorded by the filter calibration
    """
    f = interp1d(old_wav, np.squeeze(data), bounds_error=False, fill_value=0.)
    s_new = f(f_wav)
    if len(s_new.shape) == 1:
        s_new = s_new[np.newaxis, ...]
    return s_new


def get_filter_response(filename: str) -> pd.DataFrame:
    """
    read a filter response in a pd.DataFrame. first row is expected to be the
    wavelengths, first column the band names.
    :param filename: str, .csv file to parse from
    :return: pd.DataFrame, filter dataframe
    """
    df = pd.read_csv(filename, header=0, index_col=0, dtype=float)
    df.columns = df.columns.astype(float)
    return df


def get_batch_wavelengths(batch: pd.DataFrame) -> np.ndarray:
    """
    convenience function to get the wavelengths in the batch
    :param batch: pd.DataFrame, simulations batch as read by load_batch
    :return: np.ndarray, the wavelengths in the batch
    """
    return np.array(batch.reflectances.columns, dtype=float)


def load_batch(batch: Union[pd.DataFrame, str]) -> pd.DataFrame:
    """
    Loads the batch data from either a pd.DataFrame or a .csv file. Has exception handling if batch has not the
    correct data type.
    :batch: pd.DataFrame or .csv file, contains (camera) data which should be processed
    :return: pd.DataFrame, for further processing
    """
    if isinstance(batch, pd.DataFrame):
        df = batch.copy()
    elif os.path.isfile(batch):
        df = pd.read_csv(batch, header=[0, 1])
    else:
        raise IOError(
            "batch need to be either pandas DataFrame or .csv file, got: " + batch)
    return df


def load_imaging_system(filter_response: str, irradiance: str, df_w: Union[None, np.ndarray],
                        optical_system_parts: Union[None, List[str]] = None) -> ImagingSystem:
    """
    Create image system class from :class:`susi.mc.camera.ImagingSystem`
    :filter_response: str, path to pd.DataFrame or .xml file or .csv file, contains filter response data
    :irradiance: .csv or .txt file, contains irradiance or part of optical system
    :df_w: np.ndarray, array of wavelengths [m] to be interpolated to
    :optical_system_parts: None or List[str], containing path(s) to load_irradiance and reduce() for imaging_system
    :return: imaging system created with class :class:`susi.mc.camera.ImagingSystem`
    """
    F_interp = load_filter_response(filter_response, df_w)

    irradiance_values = None
    if irradiance is not None:
        irradiance_values = load_irradiance(irradiance, df_w).values
        # TODO: Why is this line added?
        # warnings.warn("irradiance passed through keyword argument, values are passed to 'w' of imaging system")

    complete_optical_system_series = None
    if optical_system_parts is not None:
        if len(optical_system_parts) > 0:
            optical_system_parts_series = [load_irradiance(
                opt_part, df_w) for opt_part in optical_system_parts]
            complete_optical_system_series = reduce(
                lambda x, y: x*y, optical_system_parts_series)
            complete_optical_system_series = complete_optical_system_series.values

    # use these to build imaging system
    # TODO: irradiance_values are stored in "white reference" in optical system, but this makes no sense
    imaging_system = ImagingSystem(
        df_w, F_interp.values, w=irradiance_values, q=complete_optical_system_series)
    return imaging_system


def load_filter_response(filter_response: Union[str, pd.DataFrame], df_w: Union[None, np.ndarray] = None) \
        -> pd.DataFrame:
    """
    Load the filter responses and transform to wavelengths specified in df_w. Is done in to_wav in
    susi.helpers.io_helpers.py. Has exception handling if batch has not the correct data type or .csv is empty.
    :filter_response: str, path to pd.DataFrame or .xml file or .csv file, contains filter response data
    :df_w: np.ndarray, array of wavelengths [m] to be interpolated to
    :return: pd.DataFrame, contains a copy of the wavelength specific filter response
    """
    if isinstance(filter_response, pd.DataFrame):
        f = filter_response.copy()
    elif os.path.isfile(filter_response):
        filename, file_extension = os.path.splitext(filter_response)
        if file_extension == '.csv':
            f = get_filter_response(filter_response)
            if f.empty:
                raise IOError(".csv file not in expected format.")
        else:
            raise ValueError(
                f"Can not identify file extension to read filter response: {filter_response}")
    else:
        raise IOError("Filter has to be either .xml, .csv or Pandas DataFrame. "
                      "got: " + filter_response)

    # map to correct wavelengths
    if df_w is not None:
        f = to_wav_df(f, df_w)
    return f


def load_irradiance(irradiance: Union[pd.DataFrame, pd.Series, str], df_w: Union[None, np.ndarray] = None) \
        -> pd.DataFrame:
    """
    Load irradiance from file (.csv pandas series, .txt spectrometer measurement) or directly from a DataFrame/ Series.
    Interpolates it to wavelengths specified in df_w. It can read light source irradiance, but is actually also used to
    read optical system parts as e.g. transmission of the laparoscope.

    :param irradiance: Union[pd.DataFrame, str], irradiance or part of optical system
    :param df_w: Union[None, np.ndarray], wavelengths [m] to be interpolated to
    :return: pd.DataFrame, contains interpolated (manipulated) irradiance
    """
    if irradiance is not None:
        if isinstance(irradiance, pd.DataFrame) or isinstance(irradiance, pd.Series):
            pass
        elif os.path.isfile(irradiance):
            # read irradiance
            filename, file_extension = os.path.splitext(irradiance)
            if file_extension == '.csv':
                irradiance = pd.read_csv(
                    irradiance, index_col=[0], squeeze=True)
            elif file_extension == '.txt':
                irradiance = get_spectrometer_measurement_series(irradiance)
            else:
                raise IOError("Irradiance could not be read, invalid file format: " +
                              file_extension + " can read .txt (spectrometer) or .csv (pandas series)")
        else:
            raise IOError("Irradiance should either be a spectrometer measurement "
                          "or pandas DataFrame. got " + irradiance)

        # map  to correct wavelengths
        if df_w is not None:
            if isinstance(irradiance, pd.Series):
                irradiance = to_wav_series(irradiance, df_w)
            elif isinstance(irradiance, pd.DataFrame):
                irradiance = to_wav_df(irradiance, df_w)
            else:
                raise IOError("file type is neither series nor dataframe")
    return irradiance


def switch_reflectances(df: pd.DataFrame, new_wavelengths: Iterable, new_reflectances: np.ndarray):
    """
    changes the "reflectances" in a pandas DataFrame (with MultiIndex). A copy of `df` is created while dropping
     column "reflectances" from axis=1 and level=0; a new DataFrame is created with the new
    reflectances and concatenated to the created copy.

    :param df: DataFrame with MultiIndex as columns. The column "reflectances" should be in axis=1 and level=0
    :param new_wavelengths: iterable with the new wavelengths to use as column names in the returned DataFrame MultiIndex
    :param new_reflectances: array with the new reflectances, length of `new_wavelengths` should be equal to `new_reflectances.shape[1]`
    :return: DataFrame with the new "reflectances"
    """
    if len(new_wavelengths) != new_reflectances.shape[1]:
        raise ValueError(f"Length of new_reflectances does not match new_reflectances: "
                         f"{len(new_wavelengths)}!={new_reflectances.shape[1]}")
    df_dropped = df.drop("reflectances", axis=1, level=0, inplace=False)
    new_cols = dict()
    for i, nw in enumerate(new_wavelengths):
        new_cols[("reflectances", nw)] = new_reflectances[:, i]
    new_df = pd.DataFrame(new_cols)
    new_df.index = df_dropped.index
    new_df = pd.concat([df_dropped, new_df], ignore_index=False, axis=1)
    return new_df


def adapt_to_camera_reflectance(batch: Union[pd.DataFrame, str], filter_response: str, irradiance: str,
                                optical_system_parts: Union[None, List[str]] = None, output: Union[None, str] = None) \
        -> pd.DataFrame:
    """
    Manipulation of loaded reflectance data within susi.mc.camera.dfmanipulations to receive reflectances at new
    wavelengths. Only difference to adapt_to_camera_color: df is copied (df=df.copy()).
    :batch: pd.DataFrame or .csv file, contains (camera) data which should be processed
    :filter_response: str, path to pd.DataFrame or .xml file or .csv file, contains filter response data
    :irradiance: .csv or .txt file, contains irradiance or part of optical system
    :optical_system_parts: None or List[str], containing path(s) to load_irradiance and reduce() for imaging_system
    :output: str or None. If str, the results are saved to a csv file indicated by "output"
    :return: pd.DataFrame, contains reflectance data at new wavelengths
    """
    df = load_batch(batch)
    df = df.copy()
    # df is misleading: get_batch_w... returns a np.ndarray, not a pd DataFrame
    df_w = get_batch_wavelengths(df)
    # build imaging system
    imaging_system = load_imaging_system(
        filter_response, irradiance, df_w, optical_system_parts)
    # get simulated camera intensities
    Cc = transform_reflectance(imaging_system, df.reflectances)
    if len(Cc.shape) == 1:
        Cc = Cc[np.newaxis, ...]
    bands_camera = np.arange(Cc.shape[1])

    df = switch_reflectances(df, bands_camera, Cc)
    if "penetration" in df.keys():
        p_adapted = transform_reflectance(imaging_system, df.penetration)
        df.drop(df["penetration"].columns, axis=1, level=1, inplace=True)
        for i, nw in enumerate(bands_camera):
            df["penetration", nw] = p_adapted[:, i]
    if output is not None:
        df.to_csv(output, index=False)
    # also return it (sometimes handy)
    return df
