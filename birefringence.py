import numpy as np
import pylab as plt
from pspy import pspy_utils, so_dict, so_spectra
import sys

def compute_chi2(Db, inv_cov, alpha):
    EB = (Db["EB"] + Db["BE"])/2
    residual = EB - 1/2 * (Db["EE"] - Db["BB"]) * np.tan(4 * alpha)
    chi2 = np.dot(residual, np.dot(inv_cov, residual))
    return chi2
    
    
d = so_dict.so_dict()
d.read_from_file(sys.argv[1])


binning_file = d["binning_file"]
lmax = d["lmax"]
surveys = d["surveys"]
type = d["type"]


mc_dir = "montecarlo"
spec_dir = "sim_spectra"
spectra = ["TT", "TE", "TB", "ET", "BT", "EE", "EB", "BE", "BB"]


_, _, lb, _ = pspy_utils.read_binning_file(binning_file, lmax)
n_bins = len(lb)

cov_EB = np.load("%s/cov_EB_all_cross.npy" % (mc_dir))


cov = {}
Db_EB = {}
Db_EE = {}
Db_BB = {}

id = 0
for id_sv1, sv1 in enumerate(surveys):
    arrays_1 = d["arrays_%s" % sv1]
    for id_ar1, ar1 in enumerate(arrays_1):
        for id_sv2, sv2 in enumerate(surveys):
            arrays_2 = d["arrays_%s" % sv2]
            for id_ar2, ar2 in enumerate(arrays_2):
                if  (id_sv1 == id_sv2) & (id_ar1 > id_ar2) : continue
                if  (id_sv1 > id_sv2) : continue

                spec_name = "%s_%s_%sx%s_%s" % (type, sv1, ar1, sv2, ar2)
                lb, Db = so_spectra.read_ps(spec_dir + "/%s_cross_00000.dat" % spec_name, spectra=spectra)

                
                cov = cov_EB[id * n_bins: (id + 1) * n_bins, id * n_bins: (id + 1) * n_bins]
                # let's only use the diagonal element (the rest is too noisy with the number of sim)
                cov = np.diag(np.diagonal(cov))
                
                inv_cov = np.linalg.inv(cov)
                
                chi2 = compute_chi2(Db, inv_cov, alpha=0)
                print(chi2/len(lb))
                
                id += 1
