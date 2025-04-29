import os
import numpy as np
from scipy.stats import pearsonr
try:
    from .sca_cy import cross_correlation
except ModuleNotFoundError:
    from sca.sca_cy import cross_correlation
import matplotlib.pyplot as plt
from scipy.interpolate import Akima1DInterpolator
from scipy.optimize import minimize_scalar

def _fisher_transform(r:float, norm_factor:float=1.12)->float:
    return 0.5*log((1 + r/norm_factor)/(1 - r/norm_factor))

def _inverse_fisher_transform(z:float, norm_factor:float=1.12)->float:
    return norm_factor*(exp(2*z) - 1)/(exp(2*z) + 1)

def _pearson_correlation(self, x:np.ndarray, y:np.ndarray)->float:
    return pearsonr(x, y)[0]

class CrossCorrelogram():
    """ 
    Cross-correlogram object, can be either classical Pearson, or SCA.

    Attributes: 
        max_shift_size (int):  Maximum shift (time lag) of time series for which correlation will be computed.
        scale_size (int, optional): Scale size parameter of SCA (length of the average short term window of correlation). Defaults to None. 
        corr_coeff_arr (np.ndarray): Array of correlogram values (correlation coefficients) for each shift (time lag).
        _interpolator: Interpolator object used to smoothen correlogram.
        _maxabs_shift (int): Shift (time lag) value of the correlogram's MaxAbs (absolute peak).
        _maxabs_value (float): Correlogram value of MaxAbs (absolute peak).
        _maxabs_is_min (bool): Whether or not MaxAbs (absolute peak) is a minimum or maximum of the correlogram.
    """
    def __init__(self, max_shift_size:int, scale_size:int=None):
        self.max_shift_size = max_shift_size # size of maximal shift of cross-correlation
        self.scale_size = scale_size # scale parameter of SCA (if None, classical CC will be computed)
        self.corr_coeff_arr = np.zeros(2*max_shift_size + 1, dtype=np.float32) # array storing values of the correlogram
        self._interpolator = None # interpolator object
        self._maxabs_shift, self._maxabs_value = -max_shift_size-1, -max_shift_size-1 # maxabs shift & value
        self._maxabs_is_min = None # whether or not maxabs is a minimum or a maximum of the correlogram

    def clear(self)->None:
        """
        Clear the correlogram object.
        """
        self.corr_coeff_arr = np.zeros(2*self.max_shift_size + 1, dtype=np.float32)
        self._interpolator = None
        self._maxabs_shift, self._maxabs_value = -self.max_shift_size-1, -self.max_shift_size-1

    def _scaled_correlation(self, x:np.ndarray, y:np.ndarray, use_fisher:bool=False)->float:
        """
        Python implementation of scaled correlation analysis (SCA) between two array of samples. For reference, see Nikolic et al. "Scaled correlation analysis: a better way to compute a cross-correlogram" (2012).

        Args:
            x (np.ndarray): First array of samples
            y (np.ndarray): Second array
            s (int): Scale/segment size
            use_fisher (bool, optional): Whether or not use Fisher transformation during SCA. Defaults to False.

        Returns:
            float: scaled correlation value
        """

        s = self.scale_size
        T = len(x) # window size
        K = T//s # number of scale segments that fit into window
        if not use_fisher:
            r_s = 0.0 # scaled correlation
            for i in range(K):
                r_s += _pearson_correlation(x[i*s:(i + 1)*s], y[i*s:(i + 1)*s])
            r_s /= K
        else:
            z = 0.0 # Fisher transform of scaled correlation
            for i in range(K):
                z += _fisher_transform(_pearson_correlation(x[i*s:(i + 1)*s], y[i*s:(i + 1)*s]))
            z /= K
            r_s = _inverse_fisher_transform(z) 

        return r_s
    
    def _compute_py(self, x:np.ndarray, y:np.ndarray, use_fisher:bool=True)->None:
        """
        Compute CC using pure python implementation (slow).

        Args:
            x (np.ndarray): first array of data points
            y (np.ndarray): second array of data points
            use_fisher (bool, optional): Whether or not use Fisher transformation during SCA. Defaults to True.
        """
        self.corr_coeff_arr = np.zeros(2*self.max_shift_size + 1, dtype=float)

        n_x = len(x)
        n_y = len(y)
        if self.scale_size is not None: # SCA
            # shifting x to the left, relative to y
            for shift in range(-self.max_shift_size, 0):
                self.corr_coeff_arr[shift + self.max_shift_size] = self._scaled_correlation(x[abs(shift):], y[:n_y - abs(shift)], use_fisher=use_fisher)
            # zero shift
            self.corr_coeff_arr[self.max_shift_size] = self._scaled_correlation(x, y, use_fisher=use_fisher)
            # shifting x to the right, relative to y
            for shift in range(1, self.max_shift_size + 1):
                self.corr_coeff_arr[shift + self.max_shift_size] = self._scaled_correlation(x[:n_x - shift], y[shift:], use_fisher=use_fisher)
        else: # CC
            # shifting x to the left, relative to y
            for shift in range(-self.max_shift_size, 0):
                self.corr_coeff_arr[shift + self.max_shift_size] = _pearson_correlation(x[abs(shift):], y[:n_y - abs(shift)])
            # zero shift
            self.corr_coeff_arr[self.max_shift_size] = _pearson_correlation(x, y)
            # shifting x to the right, relative to y
            for shift in range(1, self.max_shift_size + 1):
                self.corr_coeff_arr[shift + self.max_shift_size] = _pearson_correlation(x[:n_x - shift], y[shift:])

    def _compute_cy(self, x:np.ndarray, y:np.ndarray, use_fisher:bool=True)->None:
        """
        Compute CC using cython implementation (recommended).

        Args:
            x (np.ndarray): first array of data points
            y (np.ndarray): second array of data points
            use_fisher (bool, optional): Whether or not use Fisher transformation during SCA. Defaults to True.
        """
        self.corr_coeff_arr = np.zeros(2*self.max_shift_size + 1, dtype=np.float32)
        if self.scale_size is not None:
            cross_correlation(self.corr_coeff_arr, x, y, self.max_shift_size, self.scale_size, use_fisher)
        else:
            cross_correlation(self.corr_coeff_arr, x, y, self.max_shift_size, -1, use_fisher)

    def compute(self, x:np.ndarray, y:np.ndarray, use_fisher:bool=True, cc_method:str="C")->None:
        """
        Compute CC using one of the implementations (cython or python).

        Args:
            x (np.ndarray): first array of data points
            y (np.ndarray): second array of data points
            use_fisher (bool, optional): Whether or not use Fisher transformation during SCA. Defaults to True.
            cc_method (str, optional): Which implementation to use for the computation. Defaults to "C".

        Raises:
            ValueError: Input arrays are of uneven length
            ValueError: Length of input arrays at maximal shift during SCA is less than the scale size
            ValueError: Wrong implementation method is given
        """
        
        if len(x) != len(y):
            raise ValueError(f"Dimension mismatch between x ({len(x)}) and y ({len(y)})")
        
        if self.scale_size is not None and len(x) < self.scale_size + self.max_shift_size:
            raise ValueError(f"SCA: Length of correlation window at max shift ({len(x)} - {self.max_shift_size} = {len(x) - self.max_shift_size}) must be greater or equal than the scale/segment size ({self.scale_size})")
  
        x = x.astype(np.float32) # cython code expects float32
        y = y.astype(np.float32)
        if cc_method == "C":
            self._compute_cy(x, y, use_fisher=use_fisher)
        elif cc_method == "py":
            self._compute_py(x, y, use_fisher=use_fisher)
        else:
            raise ValueError("Only two options are supported for `cc_method` parameter: 'py' for pure python implementation, 'C' for cython (recommended)")
        
    def _find_extremas(self)->np.ndarray:
        """
        Find extremas (peaks) of the correlogram.

        Returns:
            np.ndarray: array of booleans representing whether the given point of the correlogram is an extrema or not
        """
        extremas = np.full(len(self.corr_coeff_arr), False, dtype=bool)
        for i in range(1, len(self.corr_coeff_arr) - 1):
            if (self.corr_coeff_arr[i] - self.corr_coeff_arr[i - 1])*(self.corr_coeff_arr[i + 1] - self.corr_coeff_arr[i]) < 0:
                extremas[i] = True
        return extremas

    def _set_interpolator(self, shifts:np.ndarray, values:np.ndarray):
        """
        Wrapper function for the interpolator object.

        Args:
            shifts (np.ndarray): Time shifts (x axis of the correlogram)
            values (np.ndarray): Correlation values (y axis of the correlogram)
        """
        self._interpolator = Akima1DInterpolator(shifts, values)

    def _interpolate(self, shift:float):
        if self._interpolator is None:
            raise ValueError('Akima interpolator object not set')
        return self._interpolator(shift)

    def get_maxabs(self)->tuple:
        """
        Function that determines & saves MaxAbs (absolute peak) of the interpolated correlogram.

        Raises:
            ValueError: Correlogram is not computed.

        Returns:
            tuple: MaxAbs shift, correlation value
        """
        if (self.corr_coeff_arr == 0.0).all():
            raise ValueError("Correlogram not computed.")

        if self.max_shift_size == 0: # if no shift was specified, return the only value as MaxAbs
            return self.max_shift_size, self.corr_coeff_arr[0]

        # finding maxabs on correlogram
        is_extrema = self._find_extremas()

        # check if no peaks were detected on correlogram
        if (is_extrema == False).all():
            maxabs_idx = np.argmax(np.abs(self.corr_coeff_arr)) # in that case, maxabs will be its maximal absolute value (either first or last point, because it's monotonic)
            self._maxabs_shift = maxabs_idx - self.max_shift_size
            self._maxabs_value = self.corr_coeff_arr[maxabs_idx]
            print(f'\tNOTE: No peaks found on the correlogram (it is monotonic function). Returning the maximal absolute value of the correlogram ({self._maxabs_value}) located at the maximal shift ({self._maxabs_shift}).')
            return self._maxabs_shift, self._maxabs_value

        argmax_idx = np.argmax(np.abs(self.corr_coeff_arr[is_extrema]))
        maxabs_idx = np.where(is_extrema == 1)[0][argmax_idx] # index of extrema with max abs in the correlogram
        # preparing data for interpolation
        shifts = np.arange(-self.max_shift_size, self.max_shift_size + 1, 1)
        values = self.corr_coeff_arr
        # interpolate & find minima         
        if self.corr_coeff_arr[maxabs_idx] < self.corr_coeff_arr[maxabs_idx - 1] and self.corr_coeff_arr[maxabs_idx] < self.corr_coeff_arr[maxabs_idx + 1]:
            self._maxabs_is_min = True
            self._set_interpolator(shifts, values)
            res = minimize_scalar(self._interpolate, bounds=(shifts[maxabs_idx - 1], shifts[maxabs_idx + 1]), method='bounded') # find minimum near the detected peak (maxabs)
            maxabs_hires_shift = res.x
            maxabs_hires_value = res.fun
        elif self.corr_coeff_arr[maxabs_idx] > self.corr_coeff_arr[maxabs_idx - 1] and self.corr_coeff_arr[maxabs_idx] > self.corr_coeff_arr[maxabs_idx + 1]:
            self._maxabs_is_min = False
            self._set_interpolator(shifts, -values) # if extrema is maxima, flip the function & find the minimas
            res = minimize_scalar(self._interpolate, bounds=(shifts[maxabs_idx - 1], shifts[maxabs_idx + 1]), method='bounded')
            maxabs_hires_shift = res.x
            maxabs_hires_value = -res.fun # flipping back to get maxima value

        self._maxabs_shift = maxabs_hires_shift
        self._maxabs_value = maxabs_hires_value

        return self._maxabs_shift, self._maxabs_value
   
    def plot(self, show_interp:bool=False, show_maxabs:bool=False, interp_shift_res:float=0.1):
        """
        Plot correlogram.

        Args:
            show_interp (bool, optional): Show interpolated version. Defaults to False.
            show_maxabs (bool, optional): Show maxabs on the plot. Defaults to False.
            interp_shift_res (float, optional): Resolution of the interpolated correlogram. Defaults to 0.1.

        Raises:
            Warning: MaxAbs was not determined.
        """
        shifts = np.arange(-self.max_shift_size, self.max_shift_size + 1, 1)
        plt.scatter(shifts, self.corr_coeff_arr, color="tab:blue", label="CC")
        plt.plot(shifts, np.abs(self.corr_coeff_arr), color="tab:orange", alpha=0.6, label="Abs(CC)")
        if show_interp:
            hires_shifts = np.arange(-self.max_shift_size, self.max_shift_size + 1, interp_shift_res)
            if self._maxabs_is_min:
                plt.plot(hires_shifts, self._interpolator(hires_shifts), color="tab:blue", label="SmoothCC")
            else:
                plt.plot(hires_shifts, -self._interpolator(hires_shifts), color="tab:blue", label="SmoothCC")
        if show_maxabs:
            if self._maxabs_shift == -self.max_shift_size-1 or self._maxabs_value == -self.max_shift_size-1:
                raise Warning("Cannot plot maxabs, as it wasn't computed. Calculate it with CrossCorrelogram.get_maxabs() method")
            plt.scatter([self._maxabs_shift], [self._maxabs_value], color="tab:red", alpha=0.6, s=100, label="MaxAbsVal")
        plt.xlabel("shifts [S.U.]")
        plt.grid()
        plt.legend()
