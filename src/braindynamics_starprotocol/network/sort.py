import numpy as np
import operator

def sort_distribution(data:np.ndarray)->tuple:
    """Function that performs extensive sort (sorting rows based on column values) of 2D data.

    Args:
        data (np.ndarray): Input 2D array of shape: <nr_elements>, <nr_bins>.

    Returns:
        tuple: Two arrays of sorted indices of rows from first-to-last and last-to first columns.
    """

    data_len, bin_nr = data.shape

    act_data = data[0:data_len:]

    indexed_data = np.zeros((data_len, bin_nr + 1)) # store indices in last column
    for i in range(len(act_data)):
        indexed_data[i] = np.append(act_data[i], i)

    # sort based on values
    sorted_data = np.array(sorted(indexed_data, key=operator.itemgetter(*range(bin_nr)), reverse=True)) # sorting first-to-last column
    inv_sorted_data = np.array(sorted(indexed_data, key=operator.itemgetter(*range(bin_nr-1,-1,-1)), reverse=True)) # sorting last-to-first column

    # return sorted indices only
    return sorted_data[:, bin_nr].astype(int), inv_sorted_data[:, bin_nr].astype(int)

