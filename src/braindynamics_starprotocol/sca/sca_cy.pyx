import numpy as np
cimport numpy as np
from libc.math cimport sqrt, log, exp

cdef float _pearson_correlation(float[:] x, float[:] y):
    cdef int i, n = x.shape[0]
    cdef float x_mean = 0.0, y_mean = 0.0
    cdef float xy_sum = 0.0, xx_sum = 0.0, yy_sum = 0.0
    
    for i in range(n):
        x_mean += x[i]
        y_mean += y[i]
    x_mean /= n
    y_mean /= n

    for i in range(n):
        xy_sum += (x[i] - x_mean)*(y[i] - y_mean)
        xx_sum += (x[i] - x_mean)*(x[i] - x_mean)
        yy_sum += (y[i] - y_mean)*(y[i] - y_mean)

    xy_cov = xy_sum/(n - 1)
    x_std = sqrt(xx_sum/(n - 1))
    y_std = sqrt(yy_sum/(n - 1))

    return xy_cov/x_std/y_std

cdef float _fisher_transform(float r):
    return 0.5*log((1 + r/1.12)/(1 - r/1.12))

cdef float _inverse_fisher_transform(float z):
    return 1.12*(exp(2*z) - 1)/(exp(2*z) + 1)

cdef float _scaled_correlation(float[:] x, float[:] y, int s, bint fisher_transform):
    cdef int n_x, n_y
    cdef int T, K, i

    n_x = x.shape[0]
    n_y = y.shape[0]

    T = n_x
    K = T//s
    if fisher_transform == 0:
        r_s = 0.0
        for i in range(K):
            r_s += _pearson_correlation(x[i*s:(i + 1)*s], y[i*s:(i + 1)*s])
        r_s /= K
    else:
        z = 0.0
        for i in range(K):
            z += _fisher_transform(_pearson_correlation(x[i*s:(i + 1)*s], y[i*s:(i + 1)*s]))
        z /= K
        r_s = _inverse_fisher_transform(z)
    
    return r_s

cpdef cross_correlation(float[:] corr_coeff_arr, float[:] x, float[:] y, int max_shift_size, int scale_size, bint fisher_transform):
    cdef int n_x, n_y 
    cdef int shift
    
    n_x = x.shape[0]
    n_y = y.shape[0]
    
    if scale_size > 0:
        # negative lag: shift x to the left, relative to y 
        for shift in range(-max_shift_size, 0):
            corr_coeff_arr[shift + max_shift_size] = _scaled_correlation(x[-shift:], y[:n_y + shift], scale_size, fisher_transform)
        # zero lag: no shift between x and y
        corr_coeff_arr[max_shift_size] = _scaled_correlation(x, y, scale_size, fisher_transform)
        # positive lag: shift x to the right, relative to y
        for shift in range(1, max_shift_size + 1):
            corr_coeff_arr[shift + max_shift_size] = _scaled_correlation(x[:n_x - shift], y[shift:], scale_size, fisher_transform)
    else:

        for shift in range(-max_shift_size, 0):
            corr_coeff_arr[shift + max_shift_size] = _pearson_correlation(x[-shift:], y[:n_y + shift])
       
        corr_coeff_arr[max_shift_size] = _pearson_correlation(x, y)
        
        for shift in range(1, max_shift_size + 1):
            corr_coeff_arr[shift + max_shift_size] = _pearson_correlation(x[:n_x - shift], y[shift:])
