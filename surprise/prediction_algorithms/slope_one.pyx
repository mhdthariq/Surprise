"""
Cython implementation of the SlopeOne algorithm
"""

# python imports
import numpy as np

# cython imports
cimport numpy as np
cimport cython

from .algo_base import AlgoBase

# init numpy array in cython
np.import_array()


class SlopeOne(AlgoBase):
    """A simple yet accurate collaborative filtering algorithm.

    This is a straightforward implementation of the Slope One algorithm
    :cite:`lemire2007a`.

    The prediction :math:`\\hat{r}_{ui}` is set as:

    .. math::
        \hat{r}_{ui} = \mu_u + \\frac{1}{|R_i^u|} \sum_{j \in R_i^u}
        \\text{dev}(i, j)

    where :math:`R_i^u` is the set of items rated by :math:`u` that are
    also in the neighborhood of :math:`i`.

    :math:`\\text{dev}(i, j)` is the average difference of ratings between item
    :math:`i` and item :math:`j`:

    .. math::
        \\text{dev}(i, j) = \\frac{1}{|U_{ij}|} \sum_{u \in U_{ij}} r_{ui} -
        r_{uj}

    See :ref:`User Guide <slope_one>` for more details.
    """

    def __init__(self):

        AlgoBase.__init__(self)

    def fit(self, trainset):

        AlgoBase.fit(self, trainset)

        cdef np.ndarray[np.double_t, ndim=2] dev, freq
        cdef int u, i, j
        cdef double r_ui, r_uj
        cdef list u_ratings

        dev = np.zeros((self.trainset.n_items, self.trainset.n_items),
                       np.double)
        freq = np.zeros((self.trainset.n_items, self.trainset.n_items),
                        np.int)

        for u_ratings in self.trainset.ur.values():
            for i, r_ui in u_ratings:
                for j, r_uj in u_ratings:
                    dev[i, j] += r_ui - r_uj
                    freq[i, j] += 1

        for i in range(self.trainset.n_items):
            dev[i, i] = 0
            for j in range(i + 1, self.trainset.n_items):
                if freq[i, j] > 0:
                    dev[i, j] /= freq[i, j]
                    dev[j, i] = -dev[i, j]

        self.dev = dev
        self.u_means = np.zeros(self.trainset.n_users)
        for u, u_ratings in self.trainset.ur.items():
            self.u_means[u] = np.mean([r for (_, r) in u_ratings])

        return self

    def estimate(self, u, i):

        if not (self.trainset.knows_user(u) and self.trainset.knows_item(i)):
            # This case will be handled by the default prediction method, which
            # is the global mean.
            # BTW, this is not a PredictionImpossible, because we can still
            # make a prediction.
            return self.default_prediction()

        cdef double num, den
        cdef int j
        cdef double r_uj

        # all items rated by u, with their ratings
        u_ratings = self.trainset.ur[u]
        num = 0
        den = 0
        for j, r_uj in u_ratings:
            if j != i and self.dev[i, j] != 0:
                num += (r_uj + self.dev[i, j])
                den += 1

        if den:
            return num / den
        else:
            # User u has not rated any similar item to i. We return the mean
            # of u's ratings.
            return self.u_means[u]
