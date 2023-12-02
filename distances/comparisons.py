r"""
Adapted from https://github.com/shnitzer/les-distance
"""


import abc
import numpy as np
import scipy.spatial as spat
import distances.les as les


class CompareBase:
    def __init__(self, num_models):
        self.all_distances = np.zeros((num_models, num_models))
    
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

    def compare_dataset(self, data1, data2, m_id1, m_id2):
        desc1 = self._comp_desc(data1)
        desc2 = self._comp_desc(data2)
        self.all_distances[m_id1, m_id2] = self._comp_dist(desc1, desc2)


class CompareLES(CompareBase):
    def __init__(self, num_models, sigma=2, nev=200, gamma=1e-8):
        super().__init__(num_models)
        self.sigma = sigma
        self.nev = nev
        self.gamma = gamma

    def _comp_desc(self, data):
        desc = les.les_desc_comp(data, self.sigma, self.nev, self.gamma)
        return desc

    def _comp_dist(self, desc1, desc2):
        dist = les.les_dist_comp(desc1, desc2)
        return dist


class CompareIMD(CompareBase):
    def __init__(self, num_models, T=np.logspace(-1, 1, 256), n_neighbor=5, m_lancoz=10):
        super().__init__(num_models)
        imd = __import__('msid')
        self.imd_descriptor = imd.msid.msid_descriptor

        # IMD hyperparameters
        self.T = T # Temperatures for heat kernel approx.
        self.IMD_N_NBRS = n_neighbor  # Number of neighbors in graph Laplacian
        self.M_LANCOZ = m_lancoz  # Number of Lanczos steps in SLQ

    def _comp_desc(self, data):
        desc = self.imd_descriptor(data, ts=self.T, k=self.IMD_N_NBRS, graph_builder='sparse', m=self.M_LANCOZ)
        return desc

    def _comp_dist(self, desc1, desc2):
        ct = np.exp(-2 * (self.T + 1 / self.T))
        dist = np.amax(ct * np.abs(desc1 - desc2))
        return dist


class CompareTDA(CompareBase):
    def __init__(self, num_models, bnum):
        super().__init__(num_models)
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
    def __init__(self, num_models, ngs=200):
        super().__init__(num_models)
        gs = __import__('gs')
        self.gs = gs

        self.NGS = ngs  # Tori results in Figure 1(d) are with NGS=2000, reduced here for speed

    def _comp_desc(self, data):
        desc = self.gs.rlts(data, n=self.NGS)
        return desc

    def _comp_dist(self, desc1, desc2):
        dist = self.gs.geom_score(desc1, desc2)
        return dist


class CompareGW(CompareBase):
    def __init__(self, num_models):
        super().__init__(num_models)
        self.ot = __import__('ot')

    def compare_dataset(self, data1, data2, m_id1, m_id2):
        n = data1.shape[0]
        p = self.ot.unif(n)
        q = self.ot.unif(n)

        dist1 = spat.distance.cdist(data1, data1)
        dist2 = spat.distance.cdist(data2, data2)
        self.all_distances[m_id1, m_id2] = self.ot.gromov_wasserstein2(dist1, dist2, p, q)


class CompareIMDbyLES(CompareBase):
    def __init__(self, num_models, gamma, T=np.logspace(-1, 1, 256)):
        super().__init__(num_models)
        self.T = T  # Temperatures for heat kernel approx.
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
