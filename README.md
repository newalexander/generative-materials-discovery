# materials-discovery

This repo contains files and data needed to reproduce the machine learning results of the paper "Materials discovery using generative machine learning". Further data is available at: https://doi.org/10.6084/m9.figshare.25783221.v4.


In particular, it contains a modified version of the `2023.01.10` release of the `alignn` package (https://github.com/usnistgov/alignn/tree/v2023.01.10), developed by NIST. This package is associated with the ALIGNN framework (Choudhary et al., 2021).

### License

See the `LICENSE.rst` file in `alignn/` for the license used by NIST for releasing `alignn`. 

Scripts developed by JHU/APL have the following license:

Copyright 2024 Johns Hopkins University Applied Physics Laboratory

Licensed under the Apache License, Version 2.0

### Instructions (PGCGM)

We used the pretrained PGCGM (Zhao et al., 2023) model provided in the current release (https://github.com/MilesZhao/PGCGM/tree/ab38ec6e4e6205253fb583ae0667b7a90e91cee5). Generating materials used the `create_cif.py`, `pymatgen_valid.py`, and `merge_valid.py` scripts in that repo.

### Installation (ALIGNN)

Follow these steps to get the library working on a GPU-capable machine. Note that we've commented out
the `install_requires` kwarg of `setup.py` because we're manually specifying dependencies prior to installing
`alignn`.

```
cd ...  # navigate to this directory beforehand.
conda create --name alignn python=3.8
conda activate alignn
mamba install numpy scipy=1.6.1 scikit-learn=0.22.2 matplotlib=3.4.1 pandas=1.2.3 -c conda-forge
mamba install jarvis-tools=2021.07.19 -c conda-forge
pip install tqdm pydantic==1.10.7 cif2cell==2.0.0a3 flake8 pycodestyle pydocstyle pyparsing ase
pip install torch==1.12.0+cu116 --extra-index-url https://download.pytorch.org/whl/cu116
mamba install -c dglteam "dgl=0.9"
pip install pytorch-ignite==0.5.0.dev20221024
pip install -e alignn/
```

### Contents

The two relevant scripts are:
- `scripts/train_stability.sh`: Use the MP data to train an ALIGNN model to predict decomposition enthalpy.
- `scripts/Ed_tern_pgcgm_evaluation.sh`: Use the trained ALGINN model to prediction decomposition enthalpy of generated structures.


### References

- Bartel, C. J., Trewartha, A., Wang, Q., Dunn, A., Jain, A., and Ceder, G. A critical examination of com- pound stability predictions from machine-learned for- mation energies. npj Computational Materials, 6(1): 97, Jul 2020. ISSN 2057-3960. doi: 10.1038/s41524-020-00362-y. URL https://doi.org/10.1038/s41524-020-00362-y
- Choudhary, K. and DeCost, B. Atomistic line graph neural network for improved materials property pre- dictions. npj Computational Materials, 7(1):185, Nov 2021. ISSN 2057-3960. doi: 10.1038/s41524-021-00650-1. URL https://doi.org/10.1038/s41524-021-00650-1
- Zhao, Y., Siriwardane, E. M. D., Wu, Z., Fu, N., Al- Fahdi, M., Hu, M., and Hu, J. Physics guided deep learning for generative design of crystal materials with symmetry constraints. npj Computational Materials, 9 (1):38, Mar 2023. ISSN 2057-3960. doi: 10.1038/s41524-023-00987-9. URL https://doi.org/10.1038/s41524-023-00987-9
