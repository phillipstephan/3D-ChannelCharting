# Scenario 2: Multistory Building

## How to Use the Code
The Jupyter notebooks in this repository are numbered.
You must execute them in the right order.

* `0_Dataset_RT.ipynb`: Use [Sionna RT](https://nvlabs.github.io/sionna/rt/index.html) and the 3D model provided in the `dataset` folder to create a CSI dataset. Due to the large file size, the dataset itself is not included in this git repository.
* `1_Baseline-DissimilarityMatrix.ipynb`: Compute the baseline dissimilarity matrix based on the angle-delay profile (ADP) dissimilarity metric. Learn more about it in [our tutorial](https://dichasus.inue.uni-stuttgart.de/tutorials/tutorial/dissimilarity-metric-channelcharting/).
* `2_Baseline-ChannelCharting.ipynb`: Train the baseline channel charting model based on a Siamese neural network and the dissimilarity matrix from the previous notebook.
* `3_Multistory-Classification.ipynb`: Estimate the floor index of each datapoint from the baseline channel chart using K-means clustering and CSI magnitudes.
* `4_Multistory-DissimilarityMatrix.ipynb`: Compute floor-specific dissimilarity matrices based on the estimated floor indices from the previous notebook.
* `5_Multistory-ChannelCharting.ipynb`: Train one channel chart per floor based on the floor-specific dissimilarity matrices and combine them into a multistory channel chart.

The notebooks use `scenario2_utils.py` for shared helper functions and write intermediate `.npy` and `.npz` files to the `dataset` and `results` folders. Later notebooks expect these generated files to exist.