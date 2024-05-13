import os
import pandas as pd
from sklearn.preprocessing import quantile_transform

used_columns = ["crkv_nhits100[:,0,0]",
                "crkv_nhits100[:,0,2]",
                "crkv_nhits100[:,1,2]",
                "crkv_nhits50[:,0,0]",
                "Distance between shower and track start",
                "Time difference between shower and track start",
                "E.trks.fitinf[:,0,0]",
                "E.trks.fitinf[:,0,1]",
                "E.trks.fitinf[:,0,5]",
                "E.trks.fitinf[:,0,9]",
                "Track length",
                "Track x-direction",
                "Track y-direction",
                "Track z-direction",
                "Track x-position",
                "Track y-position",
                "Track z-position",
                "Shower x-direction",
                "Shower y-direction",
                "Shower z-direction",
                "Shower x-position",
                "Shower y-position",
                "Shower z-position",
                "T.feat_Neutrino2020.cherCond_hits_trig_meanZposition",
                "Number of detector spheres with unscattered light signals",
                "Number of hits used in track reconstruction",
                "T.feat_Neutrino2020.n_hits_earlyTrig",
                "T.feat_Neutrino2020.QupMinusQdn",
                "T.sum_hits.ndoms",
                "T.sum_jppshower.n_selected_hits",
                "Inelasticity",
                "Particle name",
                "Is shower?"
]

def column_renamer(input):
    """
    Renames standard column names to something more human-readable.

    Returns a human-readable name corresponding to the input column name.
    If the input column name is not in the dictionary, returns the input column name.

    Arguments
    --------
    input : string
        Name of the column to be renamed

    Returns
    -------
    newname : string
        New name for the column
    """
    rename_dict = {
        "Unnamed: 0": "Run number",
        "angle_shfit_gandalf": "Angle between direction of shower and track",
        "distance_shfit_gandalf": "Distance between shower and track start",
        "dt_shfit_gandalf": "Time difference between shower and track start",
        "E.trks.len[:,0]": "Track length",
        "E.trks.dir.x[:,0]": "Track x-direction",
        "E.trks.dir.y[:,0]": "Track y-direction",
        "E.trks.dir.z[:,0]": "Track z-direction",
        "E.trks.pos.x[:,0]": "Track x-position",
        "E.trks.pos.y[:,0]": "Track y-position",
        "E.trks.pos.z[:,0]": "Track z-position",
        "E.trks.dir.x[:,1]": "Shower x-direction",
        "E.trks.dir.y[:,1]": "Shower y-direction",
        "E.trks.dir.z[:,1]": "Shower z-direction",
        "E.trks.pos.x[:,1]": "Shower x-position",
        "E.trks.pos.y[:,1]": "Shower y-position",
        "E.trks.pos.z[:,1]": "Shower z-position",
        "T.feat_Neutrino2020.cherCond_n_doms": "Number of detector spheres with unscattered light signals",
        "T.feat_Neutrino2020.gandalf_nHits": "Number of hits used in track reconstruction",
        "T.sum_mc_nu.by": "Inelasticity"
    }
    try:
        newname = rename_dict[input]
    except KeyError:
        newname = input
        
    return newname

def pdgid_converter(id):
    pdgid_dict = {
        12: "Electron neutrino",
        14: "Muon neutrino",
        16: "Tau neutrino",
        -12: "Anti electron neutrino",
        -14: "Anti muon neutrino",
        -16: "Anti tau neutrino"
    }
    return pdgid_dict[id]

def is_shower(row):
    id = row["pdgid"]
    is_cc = row["is_cc"]
    if((id == 14 or id == -14) and is_cc == 1.0):
        return False
    else:
        return True
    
def count_occurrences(dataframe, columns):
    """
    Displays how often each (combination of) values occurs in the specified columns.

    Arguments
    ---------
    dataframe : pd.Dataframe
        Dataframe to be analysed
    columns : iterable
        Iterable containing the names of the columns to be analysed
    """
    count_series = dataframe.groupby(columns).size()
    new_df = count_series.to_frame(name="Occurrences").reset_index()
    print(new_df)

def count_outliers(cutoff,column_name,dataframe):
    ShowerPositions = dataframe[column_name]
    CountOutliers=0
    for i in range(len(ShowerPositions)):
        pos = abs(ShowerPositions[i])
        if (pos>cutoff):
            CountOutliers += 1
    print(CountOutliers)

def clear_line():
    """Clears the current line in the console."""
    print("\r" + " " * (os.get_terminal_size().columns - 1), end = "\r")

#kinda violates single-responsibility principle ngl
def load_from_h5(filepaths):
    df_list = []
    for filepath in filepaths:
        df_list.append(pd.read_hdf(filepath))
    dataframe = pd.concat(df_list)

    dataframe.rename(column_renamer, axis="columns", inplace=True)
    dataframe["Particle name"] = dataframe.apply(lambda row: pdgid_converter(row["pdgid"]), axis=1)
    dataframe["Is shower?"] = dataframe.apply(is_shower, axis=1)

    return dataframe

def normalise_dataframe(dataframe, excluded_columns=[]):
    """
    Normalises each feature in the dataframe.

    Returns a dataframe where each column has been transformed to have zero mean and unit variance.
    Columns can be excluded from the transformation by specifying them in excluded_columns.

    Arguments
    ---------
    dataframe : pd.Dataframe
        Dataframe to be normalised
    excluded_columns : list
        List containing the names of the columns which should not be normalised

    Returns
    -------
    dataframe:
        Dataframe with normalised columns
    """
    unmodified_df = dataframe[excluded_columns]
    dataframe.drop(excluded_columns, axis=1, inplace=True)

    data = quantile_transform(dataframe, output_distribution="normal")
    dataframe = pd.DataFrame(data=data, columns=dataframe.columns)

    dataframe.reset_index(inplace=True, drop=True)
    unmodified_df.reset_index(inplace=True, drop=True)

    dataframe = pd.concat((dataframe, unmodified_df), axis=1)

    return dataframe


if(__name__ == "__main__"):
    filenames = ["neutrino11x.h5", "neutrino12x.h5", "neutrino13x.h5"]
    filepaths = [os.path.join("data", filename) for filename in filenames]

    dataframe = load_from_h5(filepaths)

    dataframe