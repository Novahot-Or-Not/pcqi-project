import os
import matplotlib.pyplot as plt
import numpy as np
from joblib import load

model_filename = r"m9.joblib"
model_filepath = os.path.join("models", model_filename)

classifier = load(model_filepath)

coefficients = classifier.coef_[0,:]
coefficients = np.absolute(coefficients)
plt.barh(classifier.feature_names_in_, coefficients)
plt.title("Absolute values of the coefficients used in the SVM classifier")
plt.show()