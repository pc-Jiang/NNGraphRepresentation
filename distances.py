from les.les import les_desc_comp, les_dist_comp

def les_dist(data1, data2, sigma=2, nev=500, gamma=1e-6):
    les_desc1 = les_desc_comp(data1, sigma=2, nev=500, gamma=1e-6)
    les_desc2 = les_desc_comp(data2, sigma=2, nev=500, gamma=1e-6)
    les_dist = les_dist_comp(les_desc1, les_desc2)