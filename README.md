# Three-Dimensional Radio Localization: A Channel Charting-Based Approach

This repository contains the source code used to produce the results presented in the paper

> <a href="https://ieeexplore.ieee.org/document/11443887">Phillip Stephan, Florian Euchner, Stephan ten Brink: "Three-Dimensional Radio Localization: A Channel Charting-Based Approach"</a>

which was presented at the 2025 59th Asilomar Conference on Signals, Systems, and Computers in Pacific Grove, CA, USA.

* A preprint is <a href="https://arxiv.org/abs/2512.15399">available on arXiv</a>.

## Structure of this Repository
This repository contains two folders:

* `Scenario1_Factory-Hall` for the factory hall scenario
* `Scenario2_Multistory-Building` for the multistory building scenario

## Prerequisites
The code is based on Python, TensorFlow, NumPy, SciPy, and Matplotlib.
The source files are provided as Jupyter notebooks, which can be opened directly on GitHub or with tools such as [JupyterLab](https://jupyter.org/).

We run our Channel Charting experiments on a JupyterHub server with NVMe storage, an AMD EPYC 7262 8-Core Processor, 64 GB RAM, and an NVIDIA GeForce RTX 4080 GPU to accelerate TensorFlow.
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
