import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from utils import load_from_h5, used_columns

filenames = ["new_neutrino11x.h5", "new_neutrino12x.h5", "new_neutrino13x.h5"]
filepaths = [os.path.join("data", filename) for filename in filenames]

print("Loading dataframe")
dataframe = load_from_h5(filepaths)


def preprocessing_data(dataframe,Emin):
    '''
    Removing unnecessary data and renaming likelihood columns.

    Arguments
    --------
    input   : pd.DataFrame
        Name of dataframe
    Emin    : int
        Lower energy threshold for data to be taken into account.

    Returns
    -------
    dataframe : pd.DataFrame
        Pre-processed dataframe only containing higher energy values.
    '''
    #remove rows with missing values
    print("Removing missing values ({} now)".format(dataframe.isnull().values.sum()))
    dataframe.dropna(inplace=True)
    print("Missing values: {}".format(dataframe.isnull().values.sum()))

    print('Only retaining relevant columns related to likelihood and energy')
    dataframe = dataframe.filter(["Track reconstruction likelyhood", "Shower reconstruction likelyhood","energy","Is shower?"])

    print(f'Dropping low-energy values below {Emin}')
    dataframe = dataframe[dataframe['energy']>Emin]
    dataframe = dataframe[dataframe["Track reconstruction likelyhood"]>0]

    return dataframe


def plotting_hist_scatter(dataframe,separated_figures):
    '''
    Plotting a scatter plot of the likelihood data track vs. shower.
    Plotting 2d histograms of track and shower likelihood.

    Arguments
    --------
    dataframe   : DataFrame
        Name of dataframe
    separated   : bool
        Set type of plot. If 'True', then plots separated plots, if 'False' plots come in two figures.
    '''
    x = dataframe["Track reconstruction likelyhood"]
    y = dataframe["Shower reconstruction likelyhood"]

    x_exp = np.exp(-x)
    y_exp = np.exp(y)

    df_shower   = dataframe[dataframe["Is shower?"]==True]
    x_shower    = df_shower["Track reconstruction likelyhood"]
    y_shower    = df_shower["Shower reconstruction likelyhood"]
    df_track    = dataframe[dataframe["Is shower?"]==False]
    x_track     = df_track["Track reconstruction likelyhood"]
    y_track     = df_track["Shower reconstruction likelyhood"]

    if separated_figures==True :
        plt.figure(num=1,figsize=(7, 4))
        plt.scatter(x,y,marker='.',c='C2',s=2)
        plt.xlabel('Track reconstruction likelihood')
        plt.ylabel('Shower reconstruction likelihood')
        plt.title(f'Scatter plot for track and shower events (unmarked) with energy > {Emin}')
        plt.tight_layout()

        plt.figure(num=2,figsize=(6, 4))
        plt.scatter(x_track,y_track,marker='.',label='Track event',s=2)
        plt.scatter(x_shower,y_shower,marker='.',label='Shower event',c='orange',s=2)
        plt.xlabel('Track reconstruction likelihood')
        plt.ylabel('Shower reconstruction likelihood')
        plt.title(f'Scatter plot for track and shower events with energy > {Emin}')
        plt.legend()
        plt.tight_layout()

        plt.figure(num=3,figsize=(6, 4))
        plt.scatter(x_track,y_track,marker='.',s=2)
        plt.xlabel('Track reconstruction likelihood')
        plt.ylabel('Shower reconstruction likelihood')
        plt.title(f'Scatter plot for track events with energy > {Emin}')
        plt.tight_layout()

        plt.figure(num=4,figsize=(6, 4))
        plt.scatter(x_shower,y_shower,marker='.',c='orange',s=2)
        plt.xlabel('Track reconstruction likelihood')
        plt.ylabel('Shower reconstruction likelihood')
        plt.title(f'Scatter plot for shower events with energy > {Emin}')
        plt.tight_layout()

        fig1 = plt.figure(num=5,figsize=(6, 4))
        h1 = plt.hist2d(x_track,y_track, bins=100,range=[[0,3000],[-2500,0]],norm='asinh',cmap='viridis')
        plt.xlabel('Track reconstruction likelihood')
        plt.ylabel('Shower reconstruction likelihood')
        plt.title(f'2d histogram, track data with energy > {Emin}')
        fig1.colorbar(h1[3])
        plt.tight_layout()

        fig2 = plt.figure(num=6,figsize=(6, 4))
        h2 = plt.hist2d(x_shower,y_shower, bins=100,range=[[0,3000],[-2500,0]],norm='asinh',cmap='viridis')
        plt.xlabel('Track reconstruction likelihood')
        plt.ylabel('Shower reconstruction likelihood')
        plt.title(f'2d histogram, shower data with energy > {Emin}')
        fig2.colorbar(h2[3])
        plt.tight_layout()

        plt.show()
    
    else:
        fig1, [[ax1, ax2], [ax3, ax4]] = plt.subplots(2,2,figsize=(15,15))

        ax1.scatter(x,y,c='C2',s=2)
        ax1.set_xlabel('Track reconstruction likelihood')
        ax1.set_ylabel('Shower reconstruction likelihood')
        ax1.set_title('Scatter plot for track and shower events (unmarked)')

        ax2.scatter(x_track,y_track,marker='.',label='Track event',s=2)
        ax2.scatter(x_shower,y_shower,marker='.',label='Shower event',c='orange',s=2)
        ax2.set_xlabel('Track reconstruction likelihood')
        ax2.set_ylabel('Shower reconstruction likelihood')
        ax2.set_title('Scatter plot for track and shower events')
        ax2.legend()

        ax3.scatter(x_track,y_track,marker='.',label='Track event',s=2)
        ax3.set_xlabel('Track reconstruction likelihood')
        ax3.set_ylabel('Shower reconstruction likelihood')
        ax3.set_title('Scatter plot for track events')
        ax3.legend()

        ax4.scatter(x_shower,y_shower,marker='.',label='Shower event',c='orange',s=2)
        ax4.set_xlabel('Track reconstruction likelihood')
        ax4.set_ylabel('Shower reconstruction likelihood')
        ax4.set_title('Scatter plot for shower events')
        ax4.legend()

        fig1.suptitle(f'Likelihood scatter plots for data with energy > {Emin}')

        plt.show()


        fig2, [[ax5,ax6], [ax7, ax8]] = plt.subplots(2,2,figsize=(15,15))

        h1 = ax5.hist2d(x,y, bins=100,range=[[0,3000],[-2500,0]],norm='asinh',cmap='viridis')
        ax5.set_xlabel('Track reconstruction likelihood')
        ax5.set_ylabel('Shower reconstruction likelihood')
        ax5.set_title('2d histogram, all data')
        fig2.colorbar(h1[3])

        ax6.scatter(x,y,marker='.',c='orange',s=2)
        ax6.set_xlabel('Track reconstruction likelihood')
        ax6.set_ylabel('Shower reconstruction likelihood')
        ax6.set_title('Scatter plot for track and shower events')
        ax6.legend()

        h3 = ax7.hist2d(x_track,y_track, bins=100,range=[[0,3000],[-2500,0]],norm='asinh',cmap='viridis')
        ax7.set_xlabel('Track reconstruction likelihood')
        ax7.set_ylabel('Shower reconstruction likelihood')
        ax7.set_title('2d histogram, track data')
        fig2.colorbar(h3[3])

        h4 = ax8.hist2d(x_shower,y_shower, bins=100,range=[[0,3000],[-2500,0]],norm='asinh',cmap='viridis')
        ax8.set_xlabel('Track reconstruction likelihood')
        ax8.set_ylabel('Shower reconstruction likelihood')
        ax8.set_title('2d histogram, shower data')
        fig2.colorbar(h4[3])

        fig2.suptitle(f'Likelihood scatter plots and histograms with energy > {Emin}')


        plt.show()



Emin = 9000
dataframe = preprocessing_data(dataframe, Emin)

plotting_hist_scatter(dataframe,separated_figures=True)