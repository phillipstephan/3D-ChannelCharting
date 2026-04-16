# Three-Dimensional Radio Localization: A Channel Charting-Based Approach

This repository contains code closely based on the implementation used to produce the results presented in the paper

> <a href="https://ieeexplore.ieee.org/document/11443887">Phillip Stephan, Florian Euchner, Stephan ten Brink: "Three-Dimensional Radio Localization: A Channel Charting-Based Approach"</a>

which was presented at the 2025 59th Asilomar Conference on Signals, Systems, and Computers in Pacific Grove, CA, USA.

* A preprint is <a href="https://arxiv.org/abs/2512.15399">available on arXiv</a>.

## Structure of this Repository
This repository contains two folders:

* `Scenario1_Factory-Hall` for the factory hall scenario
* `Scenario2_Multistory-Building` for the multistory building scenario

## Prerequisites
The notebooks and helper modules use Python, TensorFlow/Keras, NumPy, SciPy, Matplotlib, Sionna RT, scikit-learn, and tqdm.
For reference, the experiments reported here were run with Python 3.12.10, TensorFlow 2.19.0, Keras 3.10.0, NumPy 1.26.4, SciPy 1.15.2, Matplotlib 3.10.3, Sionna RT 1.0.2, scikit-learn 1.6.1, and tqdm 4.67.1.
The source code is provided primarily as Jupyter notebooks, complemented by small Python helper modules, and can be opened directly on GitHub or with tools such as [JupyterLab](https://jupyter.org/).

We run our Channel Charting experiments on a JupyterHub server with NVMe storage, an AMD EPYC 7262 8-Core Processor, 64 GB RAM, and an NVIDIA GeForce RTX 4090 GPU to accelerate TensorFlow.
All reported computation times were measured on this system.
The notebooks should also run on less powerful systems.

## Citation

```
@inproceedings{11443887,
  author = {Stephan, Phillip and Euchner, Florian and ten Brink, Stephan},
  title = {{Three-Dimensional Radio Localization: A Channel Charting-Based Approach}},
  booktitle = {2025 59th Asilomar Conference on Signals, Systems, and Computers},
  doi = {10.1109/IEEECONF67917.2025.11443887},
  pages = {880-885},
  url = {https://ieeexplore.ieee.org/document/11443887},
  year = {2025}}
```

## Other Resources
* [Christoph Studer's Channel Charting Website](https://channelcharting.github.io/)
* [Our tutorial on dissimilarity metric-based Channel Charting](https://dichasus.inue.uni-stuttgart.de/tutorials/tutorial/dissimilarity-metric-channelcharting/)
* [Our paper on dissimilarity metric-based Channel Charting](https://arxiv.org/abs/2308.09539)
