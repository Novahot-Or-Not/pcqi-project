import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
import pandas as pd
import os

from utils import column_renamer, is_shower, pdgid_converter, clear_line


def plot_density(dataframe, *, gridsize = 100, datapoint_count = 1000, xmin = 300, xmax = 600, ymin = 450, ymax = 700):
    x = dataframe[0:datapoint_count]["Track x-position"]
    y = dataframe[0:datapoint_count]["Track y-position"]

    X, Y = np.mgrid[xmin:xmax:gridsize * 1j, ymin:ymax:gridsize * 1j]
    positions = np.vstack([X.ravel(), Y.ravel()])
    values = np.vstack([x, y])
    kernel = st.gaussian_kde(values)
    Z = np.reshape(kernel(positions).T, X.shape)

    fig, ax = plt.subplots()
    ax.imshow(np.rot90(Z), cmap=plt.cm.gist_earth_r, extent=[xmin, xmax, ymin, ymax])
    #ax.plot(x, y, 'k.', markersize=2)
    ax.set_xlim([xmin, xmax])
    ax.set_ylim([ymin, ymax])
    plt.show()


if(__name__ == "__main__"):
    xmin, xmax = 300, 600
    ymin, ymax = 450, 700
    datapoint_count = 1000
    gridsize = 100
    
    filenames = ["neutrino11x.h5", "neutrino12x.h5", "neutrino13x.h5"]
    filepaths = [os.path.join("data", filename) for filename in filenames]

    print("Loading database...", end="\r")
    df_list = []
    for filepath in filepaths:
        df_list.append(pd.read_hdf(filepath))
    dataframe = pd.concat(df_list)
    
    clear_line()
    print("Renaming columns...", end="\r")
    dataframe.rename(column_renamer, axis="columns", inplace=True)
    #dataframe["Particle name"] = dataframe.apply(lambda row: pdgid_converter(row["pdgid"]), axis=1)
    dataframe["Is shower?"] = dataframe.apply(is_shower, axis=1)
    
    clear_line()
    print("Extracing muon data...", end="\r")
    df_muon = dataframe[dataframe["Is shower?"] == False]
    df_else = dataframe[dataframe["Is shower?"] == True]
    
    clear_line()
    print("Plotting density...", end="\r")
    plot_density(df_muon, gridsize=gridsize, datapoint_count=datapoint_count, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)

    clear_line()