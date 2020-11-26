import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import os
import pickle
import sys

import matplotlib.pyplot as plt
from pspy import so_dict

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
