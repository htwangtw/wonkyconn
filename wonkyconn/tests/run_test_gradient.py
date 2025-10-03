import nibabel as nib
import pandas as pd
import matplotlib.pyplot as plt

from wonkyconn.features import calculate_gradients_correlation

strategies = [
    "Baseline",
    "cCompCor",
    "Everything",
    "ICAAROMA",
    "ICAAROMACCompCor",
    "ICAAROMAGSR",
    "ICAAROMAScrubbing",
    "ICAAROMAScrubbingGSR",
    "motionParameters",
    "motionParametersGSR",
    "motionParametersScrubbing",
    "motionParametersScrubbingGSR",
    "Wang2023aCompCor",
    "Wang2023aCompCorGSR",
    "Wang2023Scrubbing",
    "Wang2023ScrubbingGSR",
    "Wang2023Simple",
    "Wang2023SimpleGSR",
]

dictionary = {}
for strategy in strategies:

    path = f"/Users/claraelkhantour/Documents/ComputeCanada/ds228/sub-pixar002/func/task-pixar/sub-pixar002_task-pixar_feature-{strategy}_atlas-schaeferCombined_desc-correlation_matrix.tsv"
    conn_matrix = pd.read_csv(path, sep="\t", header=None).to_numpy()

    atlas = nib.load("wonkyconn/data/test_data/atlas/atlas-Schaefer2018Combined_dseg.nii.gz")

    ind_aligned_gradient, group_gradients = calculate_gradients_correlation.extract_gradients(conn_matrix, atlas=atlas)

    correlation = calculate_gradients_correlation.calculate_gradients_similarity(ind_aligned_gradient, group_gradients)

    dictionary[strategy] = correlation

plt.bar(range(len(dictionary)), list(dictionary.values()), align="center")
plt.xticks(range(len(dictionary)), list(dictionary.keys()))
plt.xticks(rotation=90)
plt.tight_layout()

plt.show()
