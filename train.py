import os
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from utils import load_from_h5, used_columns, normalise_dataframe, equal_entries_df, train_test_balanced
from joblib import dump


filenames = ["neutrino11x.h5", "neutrino12x.h5", "neutrino13x.h5"]
filepaths = [os.path.join("data", filename) for filename in filenames]
parquet_filepath = os.path.join("data", "neutrino_processed.parquet")
excluded_columns = ["Is shower?", "Particle name", "Inelasticity", "is_cc"]
model_filename = "model.joblib"
model_filepath = os.path.join("models", model_filename)
equalise_columns = True
equalised_columns = ["Particle name", "is_cc"]


if(not os.path.isfile(parquet_filepath)):
    #load data
    print("Loading dataframe")
    dataframe = load_from_h5(filepaths)
    
    #exclude data which is not used
    print("Excluding unused data")
    dataframe = dataframe[used_columns]
    
    #remove rows with missing values
    print("Removing missing values ({} now)".format(dataframe.isnull().values.sum()))
    dataframe.dropna(inplace=True)
    print("Missing values: {}".format(dataframe.isnull().values.sum()))

    #drop rows to get equal amounts of data from each particle type
    if equalise_columns == True:
        print('Equalizing distribution per particle')
        dataframe = equal_entries_df(equalised_columns, dataframe, used_columns)

    #normalise data
    print("Normalising data")
    dataframe = normalise_dataframe(dataframe, excluded_columns)

    #save to parquet file
    dataframe.to_parquet(parquet_filepath)
else:
    print("Loading preprocessed data")
    dataframe = pd.read_parquet(parquet_filepath)

#divide data into training/validation sets
print("Dividing data into training and validation sets")
train_columns = [x for x in used_columns if x not in excluded_columns]
X_train, X_valid, y_train, y_valid = train_test_balanced(dataframe, equalised_columns, train_columns)
#X_train, X_valid, y_train, y_valid = train_test_split(dataframe[train_columns], dataframe["Is shower?"])

print("Training samples:\t{}".format(X_train.shape[0]))
print("Validation samples:\t{}".format(X_valid.shape[0]))

#train model
print("Training model")
classifier = LinearSVC(dual="auto")
classifier.fit(X_train, y_train)

#validate model
print("Score: {}".format(classifier.score(X_valid, y_valid)))

#save model
print("Saving model at {}".format(model_filepath))
dump(classifier, model_filepath)