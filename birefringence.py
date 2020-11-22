import numpy as np
import pylab as plt
from pspy import pspy_utils, so_dict, so_spectra
import sys
from cobaya.run import run
from getdist.mcsamples import MCSamplesFromCobaya
#from getdist import plots


def compute_chi2(Db, inv_cov, alpha):
    EB = (Db["EB"] + Db["BE"])/2
    residual = EB - 1/2 * (Db["EE"] - Db["BB"]) * np.tan(4 * alpha)
    chi2 = np.dot(residual, np.dot(inv_cov, residual))
    return chi2

def compute_sigma_alpha(Db, inv_cov):
    vec =  (Db["EE"] - Db["BB"])
    fisher = 2 * np.dot( vec, np.dot(inv_cov, vec))
    return 1/np.sqrt(fisher) * 180 / np.pi
    

    
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
print(lb)

cov_EB = np.load("%s/cov_EB_all_cross.npy" % (mc_dir))


std_mcmc_list = []
std_fisher_list = []

spec_id = 0
for id_sv1, sv1 in enumerate(surveys):
    arrays_1 = d["arrays_%s" % sv1]
    for id_ar1, ar1 in enumerate(arrays_1):
        for id_sv2, sv2 in enumerate(surveys):
            arrays_2 = d["arrays_%s" % sv2]
            for id_ar2, ar2 in enumerate(arrays_2):
                if  (id_sv1 == id_sv2) & (id_ar1 > id_ar2) : continue
                if  (id_sv1 > id_sv2) : continue

                spec_name = "%s_%s_%sx%s_%s" % (type, sv1, ar1, sv2, ar2)
                lb, Db = so_spectra.read_ps(spec_dir + "/%s_cross_00009.dat" % spec_name, spectra=spectra)
                
                
                cov = cov_EB[spec_id * n_bins: (spec_id + 1) * n_bins, spec_id * n_bins: (spec_id + 1) * n_bins]
                # let's only use the diagonal element (the rest is too noisy given the number of sim)
                cov = np.diag(np.diagonal(cov))
                inv_cov = np.linalg.inv(cov)
                
                
                # This bit is doing a mcmc fit for alpha
                
                def compute_loglike(alpha):
                    chi2 = compute_chi2(Db, inv_cov, alpha)
                    return -0.5 * chi2
                
                info = {"likelihood": {"chi2": compute_loglike},
                        "params": dict([("alpha",{"prior": {"min": -np.deg2rad(5), "max": np.deg2rad(5)},"latex": r"\alpha",},)]),
                        "sampler": {"mcmc": {"max_tries": 10 ** 8, "Rminus1_stop": 0.01, "Rminus1_cl_stop": 0.08}},}

                updated_info, sampler = run(info)
                sample = MCSamplesFromCobaya(updated_info, sampler.products()["sample"])
                
                spec_id += 1


                #alpha, chi2, _ = sample.samples[:,0], sample.samples[:,1], sample.samples[:,2]
                #alpha *= 180 / np.pi
            
               # std_mcmc = np.std(alpha)
               # std_fisher = compute_sigma_alpha(Db, inv_cov)
                
                
               # std_mcmc_list += [std_mcmc]
              #  std_fisher_list += [std_fisher]

                

#print(np.array(std_fisher_list)/np.array(std_mcmc_list))
