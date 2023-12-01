# Neural Network Representation as Graph

Author: Pengcen Jiang, YIngyu Lin

Course: DSC205 Geometry of Data, FA23@UCSD 

Instructor: Dr. Gal Mishine

This project is based on the paper: Shnitzer, Tal, et al. "Log-euclidean signatures for intrinsic distances between unaligned datasets." International Conference on Machine Learning. PMLR, 2022. [paper](https://proceedings.mlr.press/v162/shnitzer22a.html) [1] [github](https://github.com/shnitzer/les-distance)

## Instructions
Codes are built with Python version=3.11.5

Run  `pip install -r requirements.txt ` to install all the required packages

### Optional packages
Optional packages and repositories for comparisons with other algorithms:
    - [IMD](https://github.com/xgfs/imd) [2] - clone and place the `msid` folder in the current folder.
    - TDA: H0, H1 and H2 bottleneck distances, requires [persim](https://pypi.org/project/persim/)
      , [ripser](https://pypi.org/project/ripser/).
    - [GS](https://github.com/KhrulkovV/geometry-score) [3] - clone and place the `gs` folder in the current folder.\
      Requires [GUDHI](https://gudhi.inria.fr/python/latest/installation.html)
      and [Cython](https://pypi.org/project/Cython/).
    - GW [4] - requires [pot](https://pythonot.github.io/auto_examples/plot_Intro_OT.html).

## Repository structure



## References
[1] Shnitzer, Tal, et al., "Log-euclidean signatures for intrinsic distances between unaligned datasets", ICML, 2022. \
[2] Tsitsulin et al., "The Shape of Data: Intrinsic Distance for Data Distributions", ICLR, 2019.\
[3] Khrulkov and Oseledets, "Geometry score: A method for comparing generative adversarial networks", ICML, 2018.\
[4] Peyré et al., "Gromov-Wasserstein averaging of kernel and distance matrices", ICML, 2016.
