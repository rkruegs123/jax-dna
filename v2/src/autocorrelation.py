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
    import pandas as pd
    from tqdm import tqdm


    basedir = Path("/home/ryan/Documents/Harvard/research/brenner/tmp-oxdna/all-50k-autocorr")
    start_idx = 5000
    all_autocorrs = list()
    all_ks = list(range(1, 6)) + list(range(11, 16)) + list(range(20, 41))
    for k in tqdm(all_ks):
        fname = basedir / f"pdists_k{k}.pkl"
        k_dists = pickle.load(open(fname, "rb"))
        k_dists = np.array([float(v) for v in k_dists])
        k_dists = k_dists[start_idx:]
        k_dists -= np.mean(k_dists)
        k_dists_autocorr = autocorr(k_dists)
        all_autocorrs.append(k_dists_autocorr)
    mean_autocorr = np.mean(all_autocorrs, axis=0)
    plt.plot(mean_autocorr)
    plt.show()

    pdb.set_trace()



    bpath = Path("/home/ryan/Documents/Harvard/research/brenner/jaxmd-oxdna/")
    df = pd.read_csv(bpath / "distances_autocorr.txt", header=None, delim_whitespace=True)
    distances = df[2][400000:900000].values
    n = len(distances)
    window_size = 500000
    all_autocorrs = list()
    pdb.set_trace()
    assert(n % window_size == 0)
    for start_idx in tqdm(range(0, n, window_size)):
        window_distances = distances[start_idx:start_idx+window_size]
        window_distances -= np.mean(window_distances)
        window_distances_autocorr = autocorr(window_distances)
        all_autocorrs.append(window_distances_autocorr)
    mean_autocorr = np.mean(all_autocorrs, axis=0)
    plt.plot(mean_autocorr)
    plt.show()

    pdb.set_trace()


    bpath = Path("/home/ryan/Documents/Harvard/research/brenner/jaxmd-oxdna/")
    distances = pickle.load(open(bpath / "pdists_autocorrelate_0.05.pkl", "rb"))


    distances_0 = distances[:10000]
    distances_0 -= np.mean(distances_0)
    distances_0_autocorr = autocorr(distances_0)

    distances_1 = distances[10000:20000]
    distances_1 -= np.mean(distances_1)
    distances_1_autocorr = autocorr(distances_1)

    distances_2 = distances[20000:30000]
    distances_2 -= np.mean(distances_2)
    distances_2_autocorr = autocorr(distances_2)

    distances_3 = distances[30000:40000]
    distances_3 -= np.mean(distances_3)
    distances_3_autocorr = autocorr(distances_3)

    distances_4 = distances[40000:]
    distances_4 -= np.mean(distances_4)
    distances_4_autocorr = autocorr(distances_4)

    pdb.set_trace()

    plt.plot(distances_autocorr[:20000])
    plt.show()


    pdb.set_trace()

    # bpath = Path("/home/ryan/Documents/Harvard/research/brenner/tmp-oxdna/metad_2023-01-17_20-06-16")

    # centers = pickle.load(open(bpath / "centers.pkl", "rb"))

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
