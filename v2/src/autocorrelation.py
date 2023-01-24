import pdb
import numpy as np
import pickle
from pathlib import Path
import matplotlib.pyplot as plt


def autocorr(x):
    result = np.correlate(x, x,
                          mode='full'
                          # mode='valid'
    )
    return result[result.size // 2:]

if __name__ == "__main__":
    bpath = Path("/home/ryan/Documents/Harvard/research/brenner/tmp-oxdna/metad_2023-01-17_20-06-16")

    centers = pickle.load(open(bpath / "centers.pkl", "rb"))

    all_qs = centers[:, 0]
    all_qs -= np.mean(all_qs)
    all_distances = centers[:, 1]
    all_distances -= np.mean(all_distances)

    q_autocorr = autocorr(all_qs)
    distances_autocorr = autocorr(all_distances)
    pdb.set_trace()

    plt.plot(q_autocorr[:3000])
    # plt.plot(q_autocorr)
    plt.show()

    plt.clf()

    plt.plot(distances_autocorr[:5000])
    # plt.plot(distances_autocorr)
    plt.show()
