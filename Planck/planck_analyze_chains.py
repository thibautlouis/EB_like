import os
import pickle
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pspy import pspy_utils, so_dict

import planck_utils

d = so_dict.so_dict()
d.read_from_file(sys.argv[1])

chain_dir = "chains"
results = pickle.load(open(os.path.join(chain_dir, "results.pkl"), "rb"))

# Create a palette of colors given the number of spectra
colors = [plt.get_cmap("viridis", len(results))(i) for i in range(len(results))]

# Acceptance rate and R-1 plots
fig, ax = plt.subplots(1, 2, figsize=(10, 6))
ax[0].set_ylabel("acceptance rate")
ax[1].set_ylabel("$R-1$")
for i, (k, v) in enumerate(results.items()):
    v["progress"].plot(x="N", y="acceptance_rate", ax=ax[0], color=colors[i], legend=False)
    v["progress"].plot(x="N", y="Rminus1", ax=ax[1], color=colors[i], legend=False, logy=True)
plt.legend(results.keys())
plt.savefig(os.path.join(chain_dir, "progress.png"))

# Distributions of alpha posteriors
from getdist import plots

samples = [v["sample"] for k, v in results.items()]
line_args = [{"ls": "-", "color": colors[i]} for i in range(len(samples))]

g = plots.get_single_plotter(ratio=1, width_inch=6)
g.plot_1d(samples, "alpha", line_args=line_args)
plt.xlabel(r"$\alpha$ [deg.]")
plt.legend(results.keys())
plt.savefig(os.path.join(chain_dir, "posteriors.png"))

# Print mean and variance values
mean_mcmc = [result["sample"].getMeans()[0] for result in results.values()]
std_mcmc = [np.sqrt(result["sample"].getVars()[0]) for result in results.values()]

df = pd.DataFrame(
    {
        "mean MCMC": mean_mcmc,
        "std. MCMC": std_mcmc,
        "µ/σ": np.array(mean_mcmc) / np.array(std_mcmc),
    },
    index=results.keys(),
)
print(df)

# Show best fit
binning_file = d["binning_file"]
lmax = d["lmax"]
clfile = d["theoryfile"]

lth, Clth = pspy_utils.ps_lensed_theory_to_dict(
    clfile, output_type="Cl", lmax=lmax, start_at_zero=False
)

lb, Cb_EE_th = planck_utils.binning(lth, Clth["EE"], lmax, binning_file)
lb, Cb_BB_th = planck_utils.binning(lth, Clth["BB"], lmax, binning_file)
lmin, lmax = 100, 1500
idl = np.where((lb >= lmin) & (lb <= lmax))
Cb_EE_th, Cb_BB_th = Cb_EE_th[idl], Cb_BB_th[idl]

if d["use_ffp10"]:
    mc_dir = "montecarlo_ffp10_larger_bin"
else:
    mc_dir = "montecarlo"
from itertools import combinations_with_replacement as cwr

fig, axes = plt.subplots(len(results) // 2, 2, sharex=True, sharey=False, figsize=(12, 8))

for i, (f0, f1) in enumerate(cwr(d["freqs"], 2)):
    lb, Cb_EB, std_EB = np.loadtxt("%s/EB_legacy_%sx%s.dat" % (mc_dir, f0, f1), unpack=True)

    def Cb_EB_fit(alpha):
        return 1 / 2 * (Cb_EE_th - Cb_BB_th) * np.sin(4 * np.deg2rad(alpha))

    def chi2(alpha):
        return np.sum((Cb_EB - Cb_EB_fit(alpha)) ** 2 / std_EB ** 2)

    def label(alpha):
        return r"$\chi^2$/dof($\alpha=${:.2f}°) = {:.2f}/{} - PTE = {:.2f}".format(
            alpha, chi2(alpha), len(lb), stats.chi2.sf(chi2(alpha), len(lb))
        )

    from scipy import stats

    axes.flat[i].errorbar(lb, Cb_EB, std_EB, fmt=".", zorder=0, label=label(alpha=0.0))
    axes.flat[i].plot(
        lb, Cb_EB_fit(mean_mcmc[i]), "tab:red", zorder=1, label=label(alpha=mean_mcmc[i])
    )
    axes.flat[i].set_title("Planck - {}x{} GHz".format(f0, f1))
    axes.flat[i].legend(fontsize=8)

for ax in axes[-1]:
    ax.set_xlabel("$\ell$")
plt.subplots_adjust()
plt.savefig(os.path.join(chain_dir, "spectra.png"))
