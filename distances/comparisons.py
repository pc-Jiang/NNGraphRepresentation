r"""
Adapted from https://github.com/shnitzer/les-distance
"""


import abc
import numpy as np
import scipy.spatial as spat
import distances.les as les


class CompareBase:
    def __init__(self, iter_num, dict_keys):
        self.all_distances = {key: np.zeros((iter_num, 1))  for key in dict_keys}
    
    @abc.abstractmethod
    def _comp_desc(self, data):
        r"""
        Compute the algorithm's descriptors per dataset
        :param data: dataset samples organized as [samples x features]
        :return desc: descriptor vector for the dataset
        """

    @abc.abstractmethod
    def _comp_dist(self, desc1, desc2):
        r"""
        Compute the algorithm's distances a pair of dataset descriptors
        :param desc1, desc2: descriptors of two datasets
        :return dist: distance between the datasets based on the given algorithm
        """

    def compare_dataset(self, ite, data1, data2, key):
        desc1 = self._comp_desc(data1)
        desc2 = self._comp_desc(data2)
        self.all_distances[key][ite] = self._comp_dist(desc1, desc2)


class CompareLES(CompareBase):
    def __init__(self, *args):
        super().__init__(*args)
        self.sigma = 2
        self.nev = 200
        self.gamma = 1e-8

    def _comp_desc(self, data):
        desc = les.les_desc_comp(data, self.sigma, self.nev, self.gamma)
        return desc

    def _comp_dist(self, desc1, desc2):
        dist = les.les_dist_comp(desc1, desc2)
        return dist


class CompareIMD(CompareBase):
    def __init__(self, *args):
        super().__init__(*args)
        imd = __import__('msid')
        self.imd_descriptor = imd.msid.msid_descriptor

        # IMD hyperparameters
        self.T = np.logspace(-1, 1, 256)  # Temperatures for heat kernel approx.
        self.IMD_N_NBRS = 5  # Number of neighbors in graph Laplacian
        self.M_LANCOZ = 10  # Number of Lanczos steps in SLQ

    def _comp_desc(self, data):
        desc = self.imd_descriptor(data, ts=self.T, k=self.IMD_N_NBRS, graph_builder='sparse', m=self.M_LANCOZ)
        return desc

    def _comp_dist(self, desc1, desc2):
        ct = np.exp(-2 * (self.T + 1 / self.T))
        dist = np.amax(ct * np.abs(desc1 - desc2))
        return dist


class CompareTDA(CompareBase):
    def __init__(self, bnum, *args):
        super().__init__(*args)
        ripser = __import__('ripser')
        self.rips = ripser.Rips(maxdim=2)
        self.persim = __import__('persim')

        self.bnum = bnum

    def _comp_desc(self, data):
        desc = self.rips.fit_transform(data)[self.bnum]
        return desc

    def _comp_dist(self, desc1, desc2):
        dist = self.persim.bottleneck(desc1, desc2)
        return dist


class CompareGS(CompareBase):
    def __init__(self, *args):
        super().__init__(*args)
        gs = __import__('gs')
        self.gs = gs

        self.NGS = 200  # Tori results in Figure 1(d) are with NGS=2000, reduced here for speed

    def _comp_desc(self, data):
        desc = self.gs.rlts(data, n=self.NGS)
        return desc

    def _comp_dist(self, desc1, desc2):
        dist = self.gs.geom_score(desc1, desc2)
        return dist


class CompareGW:
    def __init__(self, iter_num, dict_keys):
        self.ot = __import__('ot')
        self.all_distances = {key: np.zeros((iter_num, 1)) for key in dict_keys}

    def compare_dataset(self, ite, data1, data2, key):
        n = data1.shape[0]
        p = self.ot.unif(n)
        q = self.ot.unif(n)

        dist1 = spat.distance.cdist(data1, data1)
        dist2 = spat.distance.cdist(data2, data2)
        self.all_distances[key][ite] = self.ot.gromov_wasserstein2(dist1, dist2, p, q)


class CompareIMDbyLES(CompareBase):
    def __init__(self, gamma, *args):
        super().__init__(*args)
        self.T = np.logspace(-1, 1, 256)  # Temperatures for heat kernel approx.
        self.gamma = gamma

    def _comp_desc(self, les_desc):
        r"""
        :param data: Here data should be the LES descriptor
        """
        desc = np.sum((np.exp(les_desc) - self.gamma)[:, None] ** self.T, axis=0)
        return desc

    def _comp_dist(self, desc1, desc2):
        ct = np.exp(-2 * (self.T + 1 / self.T))
        dist = np.amax(ct * np.abs(desc1 - desc2))
        return dist
