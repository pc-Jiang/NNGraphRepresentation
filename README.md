# Neural Network Representation as Graph

Author: Pengcen Jiang, Yingyu Lin

Course: DSC205 Geometry of Data, FA23@UCSD 

Instructor: Dr. Gal Mishine

This project is based on the paper: Shnitzer, Tal, et al. "Log-euclidean signatures for intrinsic distances between unaligned datasets." International Conference on Machine Learning. PMLR, 2022. [paper](https://proceedings.mlr.press/v162/shnitzer22a.html) [1] [github](https://github.com/shnitzer/les-distance)

## Instructions

Codes are built with Python version=3.11.5

Run  `pip install -r requirements.txt ` to install all the required packages

### Optional packages

Optional packages and repositories for comparisons with other algorithms:

- TDA: H0, H1 and H2 bottleneck distances, requires [persim](https://pypi.org/project/persim/) , [ripser](https://pypi.org/project/ripser/).
  
  ```
  pip install cython
  pip install ripser
  pip install persim
  ```

- [GS](https://github.com/KhrulkovV/geometry-score) [2] - clone and place the `gs` folder in the current folder.  
  Requires [GUDHI](https://gudhi.inria.fr/python/latest/installation.html) and [Cython](https://pypi.org/project/Cython/).
  
  ```
  pip install cython
  pip install gudhi
  ```

- GW [3] - requires [pot](https://pythonot.github.io/auto_examples/plot_Intro_OT.html). `pip install POT`

### Run experiments

Run `python main.py -a [exp_name]` for an experiment. Available experiments are in `experiments.py`. 

For example: 

```
python main.py -a compare_representation_across_models
```

## References

[1] Shnitzer, Tal, et al., "Log-euclidean signatures for intrinsic distances between unaligned datasets", ICML, 2022. \
[2] Khrulkov and Oseledets, "Geometry score: A method for comparing generative adversarial networks", ICML, 2018.\
[3] Peyré et al., "Gromov-Wasserstein averaging of kernel and distance matrices", ICML, 2016.
