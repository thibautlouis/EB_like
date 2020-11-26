
import numpy as np
from pspy import so_dict, pspy_utils
from itertools import combinations_with_replacement as cwr
import sys
import planck_utils
from cobaya.run import run
from getdist.mcsamples import MCSamplesFromCobaya
import pickle

d = so_dict.so_dict()
d.read_from_file(sys.argv[1])

binning_file = d["binning_file"]
freqs = d["freqs"]
lmax = d["lmax"]
clfile = d["theoryfile"]

lth, Clth = pspy_utils.ps_lensed_theory_to_dict(clfile, output_type="Cl", lmax=lmax, start_at_zero=False)

lb, Cb_EE_th = planck_utils.binning(lth, Clth["EE"], lmax, binning_file)
lb, Cb_BB_th = planck_utils.binning(lth, Clth["BB"], lmax, binning_file)


if d["use_ffp10"] == True:
    mc_dir = "montecarlo_ffp10"
else:
    mc_dir = "montecarlo"

chain_dir = "chains"
pspy_utils.create_directory(chain_dir)


freq_pairs = []
for cross in cwr(freqs, 2):
    freq_pairs += [[cross[0], cross[1]]]

lmin, lmax = 100, 1500
id = np.where((lb >= lmin) & (lb <= lmax))
Cb_EE_th, Cb_BB_th = Cb_EE_th[id], Cb_BB_th[id]

runs = {}

for i, fpair in enumerate(freq_pairs):

    f0, f1 = fpair

    spec_name = "planck_%sx%s" % (f0,f1)
    
    lb, Cb_EB, std_EB = np.loadtxt("%s/EB_legacy_%sx%s.dat" % (mc_dir, f0, f1), unpack=True)
    cov = np.diag(std_EB**2)
    inv_cov = np.linalg.inv(cov)
    
    def compute_loglike(alpha):
        residual = Cb_EB - 1 / 2 * (Cb_EE_th - Cb_BB_th) * np.sin(4 * alpha * np.pi / 180)
        chi2 = np.dot(residual, np.dot(inv_cov, residual))
        return -0.5 * chi2
    
    info = { "params": {"alpha": {"prior": {"min": -5, "max": 5}, "latex": r"\alpha"}},
            "sampler": { "mcmc": { "max_tries": 10 ** 8,"Rminus1_stop": 0.01, "Rminus1_cl_stop": 0.08, }},}
    info["likelihood"] = {"chi2": compute_loglike}

    updated_info, sampler = run(info)
    sample = MCSamplesFromCobaya(updated_info, sampler.products()["sample"])

    runs[spec_name] = {
        "progress": sampler.products()["progress"],
        "sample": sample,
    }


pickle.dump(runs, open("%s/results.pkl" % chain_dir, "wb"))

    
