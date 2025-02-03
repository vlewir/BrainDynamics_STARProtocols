import numpy as np
from cliffs_delta import cliffs_delta

def compare_ews_Cliffs_delta(A:np.ndarray, B:np.ndarray)->tuple:
	"""Function that computes Cliff's delta metric for two sets of edge weights (EWD samples). Because the EWD is a bimodal distribution, we calculate separately for the negative and for the positive parts of the distribution.

	Args:
		A (np.ndarray): First array of edge weights with shape: <number of edges>, <number of trials>
		B (np.ndarray): Second array of edge weights with shape: <number of edges>, <number of trials>

	Raises:
		ValueError: Length of the first dimension (number of edges) don't match in the two arrays.

	Returns:
		tuple: Two arrays representing Cliff's delta metric of the positive and negative parts, for each edge.
	"""
	if A.shape[0] != B.shape[0]:
		raise ValueError(f"Uneven number of rows (edges) in the input arrays: {A.shape[0]} != {B.shape[0]}")

	edge_nr = A.shape[0]

    # Cliff's delta values for the positive and negative parts of the bimodal EWD
	deltas_p = []
	deltas_n = []

	for i in range(edge_nr):
		# positive part of EWD
		pos_A = A[i][np.where(A[i] >= 0)]
		pos_B = B[i][np.where(B[i] >= 0)]
		if pos_A.size > 0 and pos_B.size > 0: # check if positive part exists
			d, _ = cliffs_delta(pos_A, pos_B) 
			deltas_p += [d]
		else:
			deltas_p += [np.nan]
		# negative part
		neg_A = A[i][np.where(A[i] < 0)]
		neg_B = B[i][np.where(B[i] < 0)]
		if neg_A.size > 0 and neg_B.size > 0: # check if negative part exists
		    d, _ = cliffs_delta(neg_A, neg_B)
		    deltas_n += [d]
		else:
		    deltas_n += [np.nan]

	deltas_p = np.asarray(deltas_p)
	deltas_n = np.asarray(deltas_n)

	return deltas_n, deltas_p

def compare_nds_Cliffs_delta(A:np.ndarray, B:np.ndarray)->np.ndarray:
	"""Function that computes Cliff's delta metric for two sets of node distances (NDD values).

	Args:
		A (np.ndarray): First array of node distances with shape: <number of nodes>, (<number of nodes> - 1)*<total number of trials 1>.
		B (np.ndarray): Second array of node distances with shape: <number of nodes>, (<number of nodes> - 1)*<total number of trials 2>.

	Raises:
		ValueError: Length of first dimension (number of nodes) don't match in the two arrays.

	Returns:
		np.ndarray: Array representing Cliff's delta metric for each node.
	"""
	if A.shape[0] != B.shape[0]:
		raise ValueError(f"Uneven number of rows (nodes) in the input arrays: {A.shape[0]} != {B.shape[0]}")

	node_nr = A.shape[0]

	deltas = []

	for i in range(node_nr):
		d, _ = cliffs_delta(A[i], B[i])
		deltas += [d]

	deltas = np.asarray(deltas)

	return deltas

def compare_news_Cliffs_delta(A:np.ndarray, B:np.ndarray)->tuple:
	"""Function that computes Cliff's delta metric for two sets of node edge weights (N-EWD values). Because the N-EWD is a bimodal distribution, we calculate separately for the negative and for the positive parts of the distribution.

	Args:
		A (np.ndarray): First array of node distances with shape: <number of nodes>, (<number of nodes> - 1)*<total number of trials 1>.
		B (np.ndarray): Second array of node distances with shape: <number of nodes>, (<number of nodes> - 1)*<total number of trials 2>.

	Raises:
		ValueError: Length of first dimension (number of nodes) don't match in the two arrays.

	Returns:
		tuple: Two arrays representing Cliff's delta metric of the positive and negative parts, for each node.
	"""
	if A.shape[0] != B.shape[0]:
		raise ValueError(f"Uneven number of rows (nodes) in the input arrays: {A.shape[0]} != {B.shape[0]}")

	node_nr = A.shape[0]

    # Cliff's delta values for the positive and negative parts of the bimodal N-EWD
	deltas_p = []
	deltas_n = []

	for i in range(node_nr):
		# positive part of N-EWD
		pos_A = A[i][np.where(A[i] >= 0)]
		pos_B = B[i][np.where(B[i] >= 0)]
		if pos_A.size > 0 and pos_B.size > 0: # check if positive part exists
			d, _ = cliffs_delta(pos_A, pos_B) 
			deltas_p += [d]
		else:
			deltas_p += [np.nan]
		# negative part
		neg_A = A[i][np.where(A[i] < 0)]
		neg_B = B[i][np.where(B[i] < 0)]
		if neg_A.size > 0 and neg_B.size > 0: # check if negative part exists
		    d, _ = cliffs_delta(neg_A, neg_B)
		    deltas_n += [d]
		else:
		    deltas_n += [np.nan]

	deltas_p = np.asarray(deltas_p)
	deltas_n = np.asarray(deltas_n)

	return deltas_n, deltas_p
