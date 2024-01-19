import pdb
import numpy as np
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns



if __name__ == "__main__":
    # bpath = Path("/home/ryan/Documents/Harvard/research/brenner/tmp-oxdna/metad_2023-01-17_20-08-15")
    bpath = Path("/home/ryan/Documents/Harvard/research/brenner/tmp-oxdna/metad_2023-01-17_20-08-44")

    centers = pickle.load(open(bpath / "centers.pkl", "rb"))

    all_qs = centers[:, 0]
    # all_qs -= np.mean(all_qs)
    all_distances = centers[:, 1]
    # all_distances -= np.mean(all_distances)

    # q_var = np.var(all_qs)
    # dist_var = np.var(all_distances)
    # q_width = np.sqrt(q_var) / 2 # should be sd
    # dist_width = np.sqrt(dist_var) / 2 # should be sd
    q_sd = np.std(all_qs)
    q_width = q_sd / 2
    dist_sd = np.std(all_distances)
    dist_width = dist_sd / 2


    pdb.set_trace()

    sns.histplot(all_qs, bins=10)
    plt.show()
    plt.clf()

    sns.histplot(all_distances, bins=10)
    plt.show()
    plt.clf()
