
import numpy as np
import pylab as plt
from pspy import so_dict, pspy_utils
from itertools import combinations_with_replacement as cwr
import sys
import planck_utils
from cobaya.run import run
from getdist.mcsamples import MCSamplesFromCobaya
import pickle
import birefringence_tools

d = so_dict.so_dict()
d.read_from_file(sys.argv[1])

binning_file = d["binning_file"]
freqs = d["freqs"]
lmax = d["lmax"]
clfile = d["theoryfile"]

lth, Clth = pspy_utils.ps_lensed_theory_to_dict(clfile, output_type="Cl", lmax=lmax, start_at_zero=False)

Cb_th = {}
lb, Cb_th["EE"] = planck_utils.binning(lth, Clth["EE"], lmax, binning_file)
lb, Cb_th["BB"]  = planck_utils.binning(lth, Clth["BB"], lmax, binning_file)


if d["use_ffp10"] == True:
    mc_dir = "montecarlo_ffp10_larger_bin"
else:
    mc_dir = "montecarlo"

chain_dir = "chains"
pspy_utils.create_directory(chain_dir)


freq_pairs = []
for cross in cwr(freqs, 2):
    freq_pairs += [[cross[0], cross[1]]]

lmin, lmax = 100, 1500
id = np.where((lb >= lmin) & (lb <= lmax))
lb, Cb_th["EE"], Cb_th["BB"] = lb[id], Cb_th["EE"][id], Cb_th["BB"][id]

runs = {}

nbins = len(lb)

Cb_th_array = np.zeros((2, nbins))
Cb_th_array[0,:] = Cb_th["EE"]
Cb_th_array[1,:] = Cb_th["BB"]

# first we read the data
cov = np.load("%s/mc_covariance_EB.npy" % mc_dir)
size_cov = cov.shape[0]
cov_EB = cov[np.int(2*size_cov/3):, np.int(2*size_cov/3): ]
cov_EB = np.diag(cov_EB.diagonal())
inv_cov = np.linalg.inv(cov_EB)

Cb_data = {}

nfreq_pairs= len(freq_pairs)
Cb_data_array = np.zeros((2, nfreq_pairs, nbins))

for id_f, fpair in enumerate(freq_pairs):
    f0, f1 = fpair
    spec_name = "planck_%sx%s" % (f0,f1)
    lb, Cb_data["EB", "%sx%s" % (f0,f1)], std_EB = np.loadtxt("%s/EB_legacy_%sx%s.dat" % (mc_dir, f0, f1), unpack=True)
    lb, Cb_data["EE", "%sx%s" % (f0,f1)], std_EE = np.loadtxt("%s/EE_legacy_%sx%s.dat" % (mc_dir, f0, f1), unpack=True)
    lb, Cb_data["BB", "%sx%s" % (f0,f1)], std_BB = np.loadtxt("%s/BB_legacy_%sx%s.dat" % (mc_dir, f0, f1), unpack=True)
    Cb_data_array[0, id_f,  :] = Cb_data["EE", "%sx%s" % (f0,f1)]
    Cb_data_array[1, id_f, :] = Cb_data["BB", "%sx%s" % (f0,f1)]

alpha = {}
alpha["100"] = 0
alpha["143"] = 0
alpha["217"] = 0
beta = 0

def compute_loglike(alpha, beta):
    vec_res = []
    for id_f, fpair in enumerate(freq_pairs):
        f0, f1 = fpair
        A = birefringence_tools.get_my_A_vector(alpha[f0], alpha[f1])
        B = birefringence_tools.get_B_vector(alpha[f0], alpha[f1], beta)
        res = Cb_data["EB", "%sx%s" % (f0,f1)] - np.dot(A, Cb_data_array[:, id_f,  :]) - np.dot(B, Cb_th_array[:,:])
        vec_res = np.append(vec_res, res)
        
    chi2 = np.dot(vec_res, np.dot(inv_cov, vec_res))
    print(chi2,len(vec_res))
    return -0.5 * chi2

print(compute_loglike(alpha, beta))
