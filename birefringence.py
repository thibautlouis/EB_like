import pickle
import sys

import numpy as np
from cobaya.run import run
from getdist.mcsamples import MCSamplesFromCobaya
from pspy import so_dict, so_mcm, so_spectra


def compute_chi2_data(Db, inv_cov, alpha):
    EB = (Db["EB"] + Db["BE"]) / 2
    residual = EB - 1 / 2 * (Db["EE"] - Db["BB"]) * np.tan(4 * alpha * np.pi / 180)
    chi2 = np.dot(residual, np.dot(inv_cov, residual))
    return chi2


def compute_chi2_theory(Db, Db_th, inv_cov, alpha):
    EB = (Db["EB"] + Db["BE"]) / 2
    residual = EB - 1 / 2 * (Db_th["EE"] - Db_th["BB"]) * np.sin(4 * alpha * np.pi / 180)
    chi2 = np.dot(residual, np.dot(inv_cov, residual))
    return chi2


def compute_sigma_alpha(Db, inv_cov):
    vec = Db["EE"] - Db["BB"]
    fisher = 4 * np.dot(vec, np.dot(inv_cov, vec))
    return 1 / np.sqrt(fisher) * 180 / np.pi


d = so_dict.so_dict()
d.read_from_file(sys.argv[1])

binning_file = d["binning_file"]
lmax = d["lmax"]
surveys = d["surveys"]
type = d["type"]

mc_dir = "montecarlo"
spec_dir = "sim_spectra"
bestfit_dir = "best_fits"
mcm_dir = "mcms"

spectra = ["TT", "TE", "TB", "ET", "BT", "EE", "EB", "BE", "BB"]

cov_EB = np.load("%s/cov_EB_all_cross.npy" % (mc_dir))

chi2_with_respect_to_theory = False

runs = {}

spec_id = 0
for id_sv1, sv1 in enumerate(surveys):
    arrays_1 = d["arrays_%s" % sv1]
    for id_ar1, ar1 in enumerate(arrays_1):
        for id_sv2, sv2 in enumerate(surveys):
            arrays_2 = d["arrays_%s" % sv2]
            for id_ar2, ar2 in enumerate(arrays_2):
                if (id_sv1 == id_sv2) & (id_ar1 > id_ar2):
                    continue
                if id_sv1 > id_sv2:
                    continue

                spec_name = "%s_%s_%sx%s_%s" % (type, sv1, ar1, sv2, ar2)
                lb, Db = so_spectra.read_ps(
                    spec_dir + "/%s_cross_00000.dat" % spec_name, spectra=spectra
                )

                if chi2_with_respect_to_theory:
                    spin_pairs = ["spin0xspin0", "spin0xspin2", "spin2xspin0", "spin2xspin2"]
                    clfile = "%s/lcdm.dat" % bestfit_dir
                    lth, Dlth = pspy_utils.ps_lensed_theory_to_dict(
                        clfile, output_type=type, lmax=lmax, start_at_zero=False
                    )
                    prefix = "%s/%s_%sx%s_%s" % (mcm_dir, sv1, ar1, sv2, ar2)
                    mbb_inv, Bbl = so_mcm.read_coupling(prefix=prefix, spin_pairs=spin_pairs)
                    Db_th = so_mcm.apply_Bbl(Bbl, Dlth, spectra=spectra)

                n_bins = len(lb)
                cov = cov_EB[
                    spec_id * n_bins : (spec_id + 1) * n_bins,
                    spec_id * n_bins : (spec_id + 1) * n_bins,
                ]
                # Let's only use the diagonal element (the rest is too noisy given the number of sim)
                cov = np.diag(np.diagonal(cov))
                inv_cov = np.linalg.inv(cov)

                # This bit is doing a mcmc fit for alpha
                def compute_loglike_data(alpha):
                    chi2 = compute_chi2_data(Db, inv_cov, alpha)
                    return -0.5 * chi2

                def compute_loglike_theory(alpha):
                    chi2 = compute_chi2_theory(Db, Db_th, inv_cov, alpha)
                    return -0.5 * chi2

                info = {
                    "params": {"alpha": {"prior": {"min": -5, "max": 5}, "latex": r"\alpha"}},
                    "sampler": {
                        "mcmc": {
                            "max_tries": 10 ** 8,
                            "Rminus1_stop": 0.01,
                            "Rminus1_cl_stop": 0.08,
                        }
                    },
                }
                if chi2_with_respect_to_theory:
                    info["likelihood"] = {"chi2": compute_loglike_theory}
                else:
                    info["likelihood"] = {"chi2": compute_loglike_data}

                updated_info, sampler = run(info)
                sample = MCSamplesFromCobaya(updated_info, sampler.products()["sample"])

                runs[spec_name] = {
                    "progress": sampler.products()["progress"],
                    "sample": sample,
                    "std_fisher": compute_sigma_alpha(Db, inv_cov),
                }

                spec_id += 1

pickle.dump(runs, open("results.pkl", "wb"))
