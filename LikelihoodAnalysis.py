import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from utils import load_from_h5, used_columns

#filenames = ["neutrino11x.h5", "neutrino12x.h5", "neutrino13x.h5"]
#filepaths = [os.path.join("data", filename) for filename in filenames]
filepaths = [os.path.join("data","newsel.h5")]

print("Loading dataframe")
dataframe = load_from_h5(filepaths)


def preprocessing_data(dataframe,Emin):
    '''
    Removing unnecessary data and renaming likelihood columns.

    Arguments
    --------
    input : DataFrame
        Name of dataframe

    Returns
    -------
    newname : int
        Lower threshold for energy values to taken into account.
    '''
    #remove rows with missing values
    print("Removing missing values ({} now)".format(dataframe.isnull().values.sum()))
    dataframe.dropna(inplace=True)
    print("Missing values: {}".format(dataframe.isnull().values.sum()))

    print('Only retaining relevant columns related to likelihood and energy')
    dataframe = dataframe.filter(["E.trks.lik[:,0]", "E.trks.lik[:,1]","energy"])

    print(f'Dropping low-energy values below {Emin}')
    dataframe = dataframe[dataframe['energy']>Emin]

    #Renaming columns
    dataframe.rename(columns = {"E.trks.lik[:,0]" : "Likelihood track" , "E.trks.lik[:,1]" : "Likelihood shower"}, inplace=True)

    return dataframe

def plotting_hist_scatter(dataframe):
    '''
    Plotting a scatter plot of the likelihood data track vs. shower and 2d histograms in chosen likelihood ranges.
    Plotting histograms of track and shower likelihood.
    '''
    x = dataframe["Likelihood track"]
    y = dataframe["Likelihood shower"]
    xmax = max(abs(x))
    ymax = max(abs(y))

    fig1, [[ax1, ax2], [ax3, ax4]] = plt.subplots(2,2,figsize=(15,15))

    ax1.scatter(x,y)
    ax1.set_xlabel('Likelihood track')
    ax1.set_ylabel('Likelihood shower')
    ax1.set_title('Scatter plot')

    h = ax2.hist2d(x,y, bins=100,cmap='viridis')
    ax2.set_xlabel('Likelihood track')
    ax2.set_ylabel('Likelihood shower')
    ax2.set_title('2d histogram')
    fig1.colorbar(h[3])

    ax3.hist(x)
    ax3.set_xlabel('Likelihood track')
    ax3.set_title('Histogram : track likelihood')

    ax4.hist(y)
    ax4.set_xlabel('Likelihood shower')
    ax4.set_title('Histogram : shower likelihood')

    fig1.suptitle(f'Likelihood histograms with energy > {Emin}')

    plt.show()


    fig2, [[ax5, ax6], [ax7, ax8]] = plt.subplots(2,2,figsize=(15,15))
    xline = np.linspace(0,xmax,1000)
    yline = np.linspace(0,-ymax,1000)

    h1 = ax5.hist2d(x,y, bins=300,cmap='viridis')
    ax5.set_xlabel('Likelihood track')
    ax5.set_ylabel('Likelihood shower')
    ax5.set_title('2d histogram')
    fig2.colorbar(h1[3])
    ax5.plot(xline,yline,color='red')

    h2 = ax6.hist2d(x,y, bins=300, range=[[0,200],[-200,0]],cmap='viridis')
    ax6.set_xlabel('Likelihood track')
    ax6.set_ylabel('Likelihood shower')
    ax6.set_title('2d histogram')
    fig2.colorbar(h2[3])
    ax6.plot(xline,yline,color='red')

    h3 = ax7.hist2d(x,y, bins=100, range=[[0,100],[-100,0]],cmap='viridis')
    ax7.set_xlabel('Likelihood track')
    ax7.set_ylabel('Likelihood shower')
    ax7.set_title('2d histogram')
    fig2.colorbar(h3[3])
    ax7.plot(xline,yline,color='red')

    h4 = ax8.hist2d(x,y, bins=100, range=[[20,75],[-125,-25]],cmap='viridis')
    ax8.set_xlabel('Likelihood track')
    ax8.set_ylabel('Likelihood shower')
    ax8.set_title('2d histogram')
    fig2.colorbar(h4[3])
    ax8.plot(xline,yline,color='red')

    fig2.suptitle(f'Likelihood histograms with energy > {Emin}')


    plt.show()



Emin = 9000
dataframe = preprocessing_data(dataframe, Emin)

plotting_hist_scatter(dataframe)


# print("Printing columns")
# for col in dataframe.columns:
#     print(col)

# print(f'The maximum absolute value of track likelihood is {xmax}')
# print(f'The maximum absolute value of shower likelihood is {ymax}')