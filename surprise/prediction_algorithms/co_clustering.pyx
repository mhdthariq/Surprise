"""
Cython implementation of the CoClustering algorithm
"""

# c imports
from libc.stdlib cimport rand, srand, RAND_MAX
from libc.math cimport log

# python imports
import numpy as np

# cython imports
cimport numpy as np
cimport cython

from .algo_base import AlgoBase
from ..utils import get_rng

# For numpy arrays, it's important to use the cimport statement, and not
# just a regular import.
# ctypedef np.int_t DTYPE_t


class CoClustering(AlgoBase):
    """A collaborative filtering algorithm based on co-clustering.

    The prediction :math:`\\hat{r}_{ui}` is set as:

    .. math::
        \hat{r}_{ui} = \\bar{C}_{uv} + (\\mu_u - \\bar{C}_u) + (\\mu_i -
        \\bar{C}_i)

    where :math:`\\bar{C}_{uv}` is the average rating of co-cluster
    :math:`C_{uv}`, :math:`\\bar{C}_u` is the average rating of user cluster
    :math:`C_u` and :math:`\\bar{C}_i` is the average rating of item cluster
    :math:`C_i`. If the user is unknown, the prediction is
    :math:`\\hat{r}_{ui} = \\mu_i`. If the item is unknown, the prediction is
    :math:`\\hat{r}_{ui} = \\mu_u`. If both are unknown, the prediction is
    :math:`\\hat{r}_{ui} = \\mu`. For details, see :cite:`George:2005`.

    Args:
        n_cltr_u(int): Number of user clusters. Default is ``3``.
        n_cltr_i(int): Number of item clusters. Default is ``3``.
        n_epochs(int): Number of iteration of the optimization loop. Default
            is ``20``.
        random_state(int, RandomState instance from numpy, or ``None``):
            Determines the RNG that will be used for initialization. If
            int, ``random_state`` will be used as a seed for a new RNG. This is
            useful to get the same initialization over multiple calls to
            ``fit()``. If RandomState instance, this same instance is used as
            RNG. If ``None``, the current RNG from numpy is used.
            Default is ``None``.
        verbose(bool): If True, prints the current epoch. Default is ``False``.
    """

    def __init__(
        self,
        n_cltr_u=3,
        n_cltr_i=3,
        n_epochs=20,
        random_state=None,
        verbose=False,
    ):

        AlgoBase.__init__(self)

        self.n_cltr_u = n_cltr_u
        self.n_cltr_i = n_cltr_i
        self.n_epochs = n_epochs
        self.random_state = random_state
        self.verbose = verbose

    def fit(self, trainset):

        AlgoBase.fit(self, trainset)

        # Make sure that n_cltr_u and n_cltr_i are not greater than the number
        # of users and items.
        self.n_cltr_u = min(self.n_cltr_u, self.trainset.n_users)
        self.n_cltr_i = min(self.n_cltr_i, self.trainset.n_items)

        # old user and item clusters
        cdef np.ndarray[int, ndim=1] old_cltr_u, old_cltr_i
        # new user and item clusters
        cdef np.ndarray[int, ndim=1] cltr_u, cltr_i

        # initialize clusters at random
        rng = get_rng(self.random_state)
        cltr_u = rng.choice(self.n_cltr_u, self.trainset.n_users, replace=True).astype(np.intc)
        cltr_i = rng.choice(self.n_cltr_i, self.trainset.n_items, replace=True).astype(np.intc)

        for epoch in range(self.n_epochs):

            if self.verbose:
                print("Processing epoch", epoch)

            # deep copy of current clusters
            old_cltr_u = np.copy(cltr_u)
            old_cltr_i = np.copy(cltr_i)

            # compute the average rating of each user cluster
            self.compute_averages(cltr_u, cltr_i)

            # assign each user to its new cluster
            self.assign_new_clusters(cltr_u, cltr_i, user_based=True)

            # re-compute the average rating of each user cluster
            self.compute_averages(cltr_u, cltr_i)

            # assign each item to its new cluster
            self.assign_new_clusters(cltr_u, cltr_i, user_based=False)

            # if clusters did not change, we reached the fix point
            if np.array_equal(cltr_u, old_cltr_u) and np.array_equal(
                cltr_i, old_cltr_i
            ):
                break

        # Now that the algorithm has converged, we have the final clusters. We
        # just need to compute the average of each cluster, user cluster and
        # item cluster.
        self.cltr_u = cltr_u
        self.cltr_i = cltr_i
        self.compute_averages(self.cltr_u, self.cltr_i)

        return self

    def estimate(self, u, i):

        est = self.trainset.global_mean
        if self.trainset.knows_user(u):
            est = self.trainset.global_mean + self.u_means[u] - self.avg_cltr_u[self.cltr_u[u]]
        if self.trainset.knows_item(i):
            est = self.trainset.global_mean + self.i_means[i] - self.avg_cltr_i[self.cltr_i[i]]

        if self.trainset.knows_user(u) and self.trainset.knows_item(i):
            est = (
                self.avg_cocltr[self.cltr_u[u], self.cltr_i[i]]
                + (self.u_means[u] - self.avg_cltr_u[self.cltr_u[u]])
                + (self.i_means[i] - self.avg_cltr_i[self.cltr_i[i]])
            )

        return est

    def compute_averages(self, int[:] cltr_u, int[:] cltr_i):
        """Compute the average rating of each user cluster, item cluster, and
        co-cluster."""

        cdef np.ndarray[double, ndim=1] avg_cltr_u, u_sum_ratings, u_n_ratings
        cdef np.ndarray[double, ndim=1] avg_cltr_i, i_sum_ratings, i_n_ratings
        cdef np.ndarray[double, ndim=2] avg_cocltr, cocltr_sum_ratings, cocltr_n_ratings
        cdef int u, i, r, cu, ci

        # The average rating of each user cluster
        avg_cltr_u = np.zeros(self.n_cltr_u)
        u_sum_ratings = np.zeros(self.n_cltr_u)
        u_n_ratings = np.zeros(self.n_cltr_u)

        # The average rating of each item cluster
        avg_cltr_i = np.zeros(self.n_cltr_i)
        i_sum_ratings = np.zeros(self.n_cltr_i)
        i_n_ratings = np.zeros(self.n_cltr_i)

        # The average rating of each co-cluster
        avg_cocltr = np.zeros((self.n_cltr_u, self.n_cltr_i))
        cocltr_sum_ratings = np.zeros((self.n_cltr_u, self.n_cltr_i))
        cocltr_n_ratings = np.zeros((self.n_cltr_u, self.n_cltr_i))

        for u, u_ratings in self.trainset.ur.items():
            for i, r in u_ratings:
                cu = cltr_u[u]
                ci = cltr_i[i]
                u_sum_ratings[cu] += r
                u_n_ratings[cu] += 1
                i_sum_ratings[ci] += r
                i_n_ratings[ci] += 1
                cocltr_sum_ratings[cu, ci] += r
                cocltr_n_ratings[cu, ci] += 1

        for cu in range(self.n_cltr_u):
            if u_n_ratings[cu]:
                avg_cltr_u[cu] = u_sum_ratings[cu] / u_n_ratings[cu]
            else:
                avg_cltr_u[cu] = self.trainset.global_mean

        for ci in range(self.n_cltr_i):
            if i_n_ratings[ci]:
                avg_cltr_i[ci] = i_sum_ratings[ci] / i_n_ratings[ci]
            else:
                avg_cltr_i[ci] = self.trainset.global_mean

        for cu in range(self.n_cltr_u):
            for ci in range(self.n_cltr_i):
                if cocltr_n_ratings[cu, ci]:
                    avg_cocltr[cu, ci] = (
                        cocltr_sum_ratings[cu, ci] / cocltr_n_ratings[cu, ci]
                    )
                else:
                    avg_cocltr[cu, ci] = self.trainset.global_mean

        self.u_means = np.zeros(self.trainset.n_users)
        for u, u_ratings in self.trainset.ur.items():
            self.u_means[u] = np.mean([r for (_, r) in u_ratings])

        self.i_means = np.zeros(self.trainset.n_items)
        for i, i_ratings in self.trainset.ir.items():
            self.i_means[i] = np.mean([r for (_, r) in i_ratings])

        self.avg_cltr_u = avg_cltr_u
        self.avg_cltr_i = avg_cltr_i
        self.avg_cocltr = avg_cocltr

        return self

    def assign_new_clusters(self, int[:] cltr_u, int[:] cltr_i, bint user_based):
        """Assign each user (or item) to its new cluster."""

        cdef int x, y, r, best_c, c
        cdef np.ndarray[double, ndim=1] errs

        if user_based:
            n_x = self.trainset.n_users
            n_c = self.n_cltr_u
            xr = self.trainset.ur
            cltr_y = cltr_i
        else:
            n_x = self.trainset.n_items
            n_c = self.n_cltr_i
            xr = self.trainset.ir
            cltr_y = cltr_u

        for x in range(n_x):
            errs = np.zeros(n_c)
            for c in range(n_c):
                # compute the error when x is in cluster c
                for y, r in xr[x]:
                    cy = cltr_y[y]
                    if user_based:
                        pred = (
                            self.avg_cocltr[c, cy]
                            + (self.u_means[x] - self.avg_cltr_u[c])
                            + (self.i_means[y] - self.avg_cltr_i[cy])
                        )
                    else:  # item_based
                        pred = (
                            self.avg_cocltr[cy, c]
                            + (self.u_means[y] - self.avg_cltr_u[cy])
                            + (self.i_means[x] - self.avg_cltr_i[c])
                        )

                    errs[c] += (r - pred) ** 2

            best_c = np.argmin(errs)
            if user_based:
                cltr_u[x] = best_c
            else:
                cltr_i[x] = best_c

        return self
