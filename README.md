# Rethinking minimax-fairness

Code for our ICML 2023 paper 'When do ERM and Minimax-fair Learning Coincide?'

---

## Installation

1. Install Anaconda environment

2. Install Python packages
```
pip install folktables xgboost netcal torch seaborn
```

3. Install package for multilayer perceptron models
```
pip install rtdl libzero
```

4. Install autogluon. Instructions to install are at [autogluon](https://auto.gluon.ai/stable/install.html) website

5. Install folktables package from source code in the folder named `folktables`
```
cd folktables
pip install -r requirements.txt
```
This code differs from the one in the GitHub [repo](https://github.com/socialfoundations/folktables) in the `df_to_pandas` function in `folktables/folktables.py`
We drop the first level of categorical variables after making the dummies

1. (Optional) Install package to pretty print tables in Jupyter
```
pip install dataframe-image
```

---

## Datasets

1. Download a part of the datasets following the instructions in this GitHub [repo](https://github.com/amazon-science/minimax-fair#datasets) for the Minimax Group Fairness [paper](https://arxiv.org/abs/2011.03108)

2. Save all of these datasets in a folder named `Datasets`

3. Rest of the datasets except eICU are downloaded automatically by the scripts 

4. Scripts to extract and download eICU dataset are at this GitHub [repo](https://github.com/alistairewj/icu-model-transfer). Accessing the data requires completing an online training course and requesting access through the PhysioNet website. Details of getting access are at this [website](https://eicu-crd.mit.edu/gettingstarted/access/)

---

## Run instructions
Run `run.sh` with the id of the dataset in the list `Dataset().list_datasets()` in `main.py`

For example, run the following script for the first dataset id
```
bash run.sh 1
```
The script runs all models for the given dataset once including and once excluding the group feature

---

## Credits (also see [THIRD-PARTY-LICENSES](https://github.com/amazon-science/rethinking-minimax-fairness/blob/main/THIRD-PARTY-LICENSES))
This repo contains code modified from the following GitHub repos

1. [folktables](https://github.com/socialfoundations/folktables), MIT license

Code included in folder `folktables` and the file `folktables_helper.py`

2. [minimax-fair](https://github.com/amazon-science/minimax-fair), Apache 2.0 license

Code included in folder `src` and the files `dataset_mapping.py, prepare_datasets.py`

3. [active-sampling-for-minmax-fairness](https://github.com/amazon-science/active-sampling-for-minmax-fairness), Apache 2.0 license

Code included in folder `algorithms` and the file `prepare_datasets.py`

---

## Licenses

| Software    | License |
| -------- | ------- |
| folktables  | [MIT License](https://github.com/socialfoundations/folktables/blob/main/LICENSE.txt)    |
| minimax-fair | [Apache License 2.0](https://github.com/amazon-science/minimax-fair/blob/main/LICENSE)     |
| active-sampling-for-minmax-fairness | [Apache License 2.0](https://github.com/amazon-science/active-sampling-for-minmax-fairness/blob/main/LICENSE)     |
| xgboost | [Apache License 2.0](https://github.com/dmlc/xgboost/blob/master/LICENSE)     |
| autogluon | [Apache License 2.0](https://github.com/autogluon/autogluon/blob/master/LICENSE)     |
| netcal    | [Apache License 2.0](https://github.com/EFS-OpenSource/calibration-framework/blob/main/LICENSE.txt)    |
| torch    | [Modified BSD 3-Clause](https://github.com/pytorch/pytorch/blob/main/LICENSE)    |
| seaborn    | [BSD 3-Clause](https://github.com/mwaskom/seaborn/blob/master/LICENSE.md)    |
| pandas    | [BSD 3-Clause](https://github.com/pandas-dev/pandas/blob/main/LICENSE)    |
| numpy    | [BSD 3-Clause](https://github.com/numpy/numpy/blob/main/LICENSE.txt)    |
| matplotlib    | [PSF License](https://matplotlib.org/stable/users/project/license.html)    |
| scikit-learn    | [BSD 3-Clause](https://github.com/scikit-learn/scikit-learn/blob/main/COPYING)    |
| rtdl    | [MIT License](https://github.com/Yura52/rtdl/blob/main/LICENSE)    |
| libzero    | [MIT License](https://github.com/Yura52/delu/blob/main/LICENSE)    |
| dataframe-image    | [MIT License](https://github.com/dexplo/dataframe_image/blob/master/LICENSE)    |

---


## Troubleshooting known errors

### libomp.dylib related error if running on Macbook M1

```
OMP: Error #15: Initializing libomp.dylib, but found libomp.dylib already initialized.
OMP: Hint This means that multiple copies of the OpenMP runtime have been linked into the program. That is dangerous, since it can degrade performance or cause incorrect results. The best thing to do is to ensure that only a single OpenMP runtime is linked into the process, e.g. by avoiding static linking of the OpenMP runtime in any library. As an unsafe, unsupported, undocumented workaround you can set the environment variable KMP_DUPLICATE_LIB_OK=TRUE to allow the program to continue to execute, but that may cause crashes or silently produce incorrect results. For more information, please see http://openmp.llvm.org/
```

Try re-running the code after running command on terminal as suggested above
```
export KMP_DUPLICATE_LIB_OK=TRUE
```
