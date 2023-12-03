# Cartesian Product - 6x faster than itertools.product - 10x less memory

## pip install cythoncartesian

### Tested against Windows / Python 3.11 / Anaconda


## Cython (and a C/C++ compiler) must be installed



```python
Generate the Cartesian product of input iterables.

Parameters:
-----------
*args : list of iterable
	Input iterables for which the Cartesian product will be computed.
outputdtype (optional) :
	dtype of output array
Returns:
--------
numpy.ndarray
	2D array containing the Cartesian product of the input iterables.

Notes:
------
This function efficiently computes the Cartesian product of the input iterables
using Cython implementation. It outperforms the equivalent functionality provided
by itertools.product, and returns a NumPy array (not a list of tuples like itertools.product).

Examples:
---------
	from cythoncartesian import cartesian_product

	# Mem usage 2 GB
	# Out[4]:
	# array([[0, 0, 0, ..., 0, 0, 0],
	#        [1, 0, 0, ..., 0, 0, 0],
	#        [2, 0, 0, ..., 0, 0, 0],
	#        ...,
	#        [5, 7, 7, ..., 7, 7, 7],
	#        [6, 7, 7, ..., 7, 7, 7],
	#        [7, 7, 7, ..., 7, 7, 7]], dtype=uint8)
	# %timeit dataresults=cartesian_product(*args2)
	# 2.65 s ± 163 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
	# dataresults.shape
	# Out[6]: (134217728, 9)
	

	# itertools.product
	# Mem usage 16 GB

	# import itertools
	# %timeit (list(itertools.product(*args2)))
	# 11.5 s ± 203 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)


	# --------------------------------------------------------------------------
	# Mem usage 1.2 GB
	# args = [[411, 231.33, 4342, 12341, 1.142, 1.33, 13],
	#         [34, 231.33, 4132, 1231],
	#          [14, 44, 23454.1, .1, 23, 1],
	#          [9, 12, 1, 3, 32, 23, 21, 31],
	#          [1114, 44, 23454.1, .1, 23, 1],
	#         ]+[list(range(6)),list(range(3)),list(range(3)),list(range(3))
	#     ,list(range(3))]+[list(range(6))]
	# %timeit dataresults=cartesian_product(*args)
	# 621 ms ± 46.9 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)


	# Mem usage 4 GB
	# import itertools
	# %timeit (list(itertools.product(*args)))
	# 2.13 s ± 26.4 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)

```