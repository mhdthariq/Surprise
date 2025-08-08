"""
Cython implementation of the ALS and SGD optimization methods.
"""

# python imports
import numpy as np

# cython imports
cimport numpy as np
cimport cython

# numpy compatibility
from numpy cimport npy_intp

# define numpy types for compatibility
ctypedef np.float64_t DTYPE_t
DTYPE = np.float64

# init numpy array in cython
np.import_array()


@cython.boundscheck(False)
def als(algo):

    cdef np.ndarray[DTYPE_t] bu, bi
    cdef int u, i, epoch
    cdef double r, dev_u, dev_i
    cdef int n_epochs, reg_u, reg_i

    trainset = algo.trainset
    bsl_options = algo.bsl_options

    n_epochs = bsl_options.get('n_epochs', 10)
    reg_u = bsl_options.get('reg_u', 15)
    reg_i = bsl_options.get('reg_i', 10)

    bu = np.zeros(trainset.n_users, np.double)
    bi = np.zeros(trainset.n_items, np.double)

    for epoch in range(n_epochs):
        # compute new user biases
        for u in range(trainset.n_users):
            dev_u = 0
            for i, r in trainset.ur[u]:
                dev_u += r - trainset.global_mean - bi[i]
            bu[u] = dev_u / (reg_u + len(trainset.ur[u]))

        # compute new item biases
        for i in range(trainset.n_items):
            dev_i = 0
            for u, r in trainset.ir[i]:
                dev_i += r - trainset.global_mean - bu[u]
            bi[i] = dev_i / (reg_i + len(trainset.ir[i]))

    return bu, bi


@cython.boundscheck(False)
def sgd(algo):

    cdef np.ndarray[DTYPE_t] bu, bi
    cdef int u, i, epoch
    cdef double r, err
    cdef int n_epochs, reg, learning_rate

    trainset = algo.trainset
    bsl_options = algo.bsl_options

    n_epochs = bsl_options.get('n_epochs', 20)
    reg = bsl_options.get('reg', 0.02)
    learning_rate = bsl_options.get('learning_rate', 0.005)

    bu = np.zeros(trainset.n_users, np.double)
    bi = np.zeros(trainset.n_items, np.double)

    for epoch in range(n_epochs):
        for u, i, r in trainset.all_ratings():
            err = r - (trainset.global_mean + bu[u] + bi[i])
            bu[u] += learning_rate * (err - reg * bu[u])
            bi[i] += learning_rate * (err - reg * bi[i])

    return bu, bi


# Create aliases for backward compatibility
baseline_als = als
baseline_sgd = sgd
