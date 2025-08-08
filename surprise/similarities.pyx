"""
Cython implementation of the similarity measures.
"""

# python imports
import numpy as np

# cython imports
cimport numpy as np
cimport cython

# numpy compatibility - import these first
from numpy cimport npy_intp
np.import_array()

# define numpy types for compatibility
ctypedef np.float64_t DTYPE_t
DTYPE = np.float64

# Verify numpy compatibility
cdef extern from "numpy/arrayobject.h":
    int NPY_VERSION_MAJOR "NPY_VERSION_MAJOR"
    int NPY_VERSION_MINOR "NPY_VERSION_MINOR"

# Check numpy version compatibility at module init
def _check_numpy_compatibility():
    major, minor = np.__version__.split('.')[:2]
    major, minor = int(major), int(minor)
    # Support numpy 1.21+ and numpy 2.x
    return (major == 1 and minor >= 21) or (major >= 2)

# Perform the check
if not _check_numpy_compatibility():
    raise ImportError(f"Numpy version {np.__version__} is not compatible. Please use numpy >= 1.21.0")


@cython.boundscheck(False)
def cosine(int n_x, dict yr, int min_support=1):

    cdef np.ndarray[DTYPE_t, ndim=2] sim
    cdef np.ndarray[DTYPE_t, ndim=1] norms
    cdef int x, y, i, nb_common, i_x, i_y
    cdef list x_ratings, y_ratings
    cdef double r_x, r_y

    sim = np.zeros((n_x, n_x), dtype=DTYPE)
    norms = np.zeros(n_x, dtype=DTYPE)

    for x in range(n_x):
        for i, r_x in yr.get(x, []):
            norms[x] += r_x * r_x
        norms[x] = np.sqrt(norms[x])

    for x in range(n_x):
        for y in range(x, n_x):

            # number of common items
            # This is faster than len(set(yr[x]) & set(yr[y]))
            nb_common = 0
            i_x = 0
            i_y = 0
            x_ratings = yr.get(x, [])
            y_ratings = yr.get(y, [])

            while i_x < len(x_ratings) and i_y < len(y_ratings):
                if x_ratings[i_x][0] < y_ratings[i_y][0]:
                    i_x += 1
                elif x_ratings[i_x][0] > y_ratings[i_y][0]:
                    i_y += 1
                else:  # same item. We can compute the dot product
                    nb_common += 1
                    sim[x, y] += x_ratings[i_x][1] * y_ratings[i_y][1]
                    i_x += 1
                    i_y += 1

            if nb_common >= min_support:
                if norms[x] != 0 and norms[y] != 0:
                    sim[x, y] /= norms[x] * norms[y]
            else:
                sim[x, y] = 0

            sim[y, x] = sim[x, y]

    return sim


@cython.boundscheck(False)
def msd(int n_x, dict yr, int min_support=1):

    cdef np.ndarray[DTYPE_t, ndim=2] sim
    cdef int x, y, nb_common, i_x, i_y
    cdef list x_ratings, y_ratings
    cdef double r_x, r_y, msd

    sim = np.zeros((n_x, n_x), dtype=DTYPE)

    for x in range(n_x):
        for y in range(x, n_x):

            # number of common items
            # This is faster than len(set(yr[x]) & set(yr[y]))
            nb_common = 0
            msd = 0
            i_x = 0
            i_y = 0
            x_ratings = yr.get(x, [])
            y_ratings = yr.get(y, [])

            while i_x < len(x_ratings) and i_y < len(y_ratings):
                if x_ratings[i_x][0] < y_ratings[i_y][0]:
                    i_x += 1
                elif x_ratings[i_x][0] > y_ratings[i_y][0]:
                    i_y += 1
                else:  # same item
                    nb_common += 1
                    msd += (x_ratings[i_x][1] - y_ratings[i_y][1])**2
                    i_x += 1
                    i_y += 1

            if nb_common >= min_support:
                sim[x, y] = 1 / (msd / nb_common + 1)
            else:
                sim[x, y] = 0

            sim[y, x] = sim[x, y]

    return sim


@cython.boundscheck(False)
def pearson(int n_x, dict yr, int min_support=1):

    cdef np.ndarray[DTYPE_t, ndim=2] sim
    cdef int x, y, nb_common, i_x, i_y
    cdef list x_ratings, y_ratings, common_ratings
    cdef double r_x, r_y, num, den_x, den_y, x_mean, y_mean

    sim = np.zeros((n_x, n_x), dtype=DTYPE)

    for x in range(n_x):
        for y in range(x, n_x):

            num = 0
            den_x = 0
            den_y = 0
            nb_common = 0

            # This is faster than using set()
            i_x = 0
            i_y = 0
            x_ratings = yr.get(x, [])
            y_ratings = yr.get(y, [])
            common_ratings = []

            while i_x < len(x_ratings) and i_y < len(y_ratings):
                if x_ratings[i_x][0] < y_ratings[i_y][0]:
                    i_x += 1
                elif x_ratings[i_x][0] > y_ratings[i_y][0]:
                    i_y += 1
                else:  # same item
                    common_ratings.append((x_ratings[i_x][1], y_ratings[i_y][1]))
                    i_x += 1
                    i_y += 1

            nb_common = len(common_ratings)
            if nb_common >= min_support:
                x_ratings_common = [r[0] for r in common_ratings]
                y_ratings_common = [r[1] for r in common_ratings]
                x_mean = np.mean(x_ratings_common)
                y_mean = np.mean(y_ratings_common)

                for r_x, r_y in common_ratings:
                    num += (r_x - x_mean) * (r_y - y_mean)
                    den_x += (r_x - x_mean)**2
                    den_y += (r_y - y_mean)**2

                if den_x != 0 and den_y != 0:
                    sim[x, y] = num / (np.sqrt(den_x) * np.sqrt(den_y))

            sim[y, x] = sim[x, y]

    return sim


@cython.boundscheck(False)
def pearson_baseline(int n_x, dict yr, int min_support, double global_mean,
                     np.ndarray[DTYPE_t] bx, np.ndarray[DTYPE_t] by,
                     int shrinkage=100):

    cdef np.ndarray[DTYPE_t, ndim=2] sim
    cdef int x, y, nb_common, i_x, i_y
    cdef list x_ratings, y_ratings
    cdef double r_x, r_y, num, den_x, den_y, dev_x, dev_y

    sim = np.zeros((n_x, n_x), dtype=DTYPE)

    for x in range(n_x):
        for y in range(x, n_x):

            num = 0
            den_x = 0
            den_y = 0
            nb_common = 0

            # This is faster than using set()
            i_x = 0
            i_y = 0
            x_ratings = yr.get(x, [])
            y_ratings = yr.get(y, [])

            while i_x < len(x_ratings) and i_y < len(y_ratings):
                if x_ratings[i_x][0] < y_ratings[i_y][0]:
                    i_x += 1
                elif x_ratings[i_x][0] > y_ratings[i_y][0]:
                    i_y += 1
                else:  # same item
                    nb_common += 1
                    # deviation from baseline
                    dev_x = x_ratings[i_x][1] - (global_mean + bx[x] +
                                                by[x_ratings[i_x][0]])
                    dev_y = y_ratings[i_y][1] - (global_mean + bx[y] +
                                                by[y_ratings[i_y][0]])
                    num += dev_x * dev_y
                    den_x += dev_x**2
                    den_y += dev_y**2
                    i_x += 1
                    i_y += 1

            if nb_common >= min_support and den_x != 0 and den_y != 0:
                sim[x, y] = (num / (np.sqrt(den_x) * np.sqrt(den_y)) *
                             nb_common / (nb_common + shrinkage))

            sim[y, x] = sim[x, y]

    return sim
