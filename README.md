# Beyond Reconstruction Fairness: Evaluating the Downstream Effects of Fair PCA Preprocessing

This repository contains the code used for the bachelor thesis *Beyond Reconstruction Fairness: Evaluating the Downstream Effects of Fair PCA Preprocessing*. The project evaluates Standard PCA, Fair PCA, and LAFTR as preprocessing methods for the COMPAS and Adult datasets. The experiments assess both group-specific reconstruction performance and downstream binary-classification outcomes.

Please refer to the uploaded `.csv` files in the `experiments/results` folder for the experimental results. Please refer to the uploaded `.png` files in the `figures/reconstruction` folder for the reconstruction figures included in the research paper. To reproduce the results reported in this research, please refer to the `notebooks` module.

## Quick demo: COMPAS data

To run a quick demo for the reconstruction experiment on COMPAS data, please run the following command:

```bash
python experiments/run_reconstruction.py --dataset COMPAS --experiment standard
```

To run a quick demo for the classification experiment on COMPAS data using the logistic regression classifier, please run the following command:

```bash
python experiments/run_classification.py --dataset COMPAS --classifier log_reg
```

To run a more extensive demo for the classification experiment on COMPAS data using all implemented classifiers, please run the following command:

```bash
python experiments/run_classification.py --dataset COMPAS --classifier all
```

## Data

The data found in this repository was retrieved from the following sources:

- COMPAS: https://github.com/propublica/compas-analysis
- Adult: https://archive.ics.uci.edu/dataset/2/adult

## References

This repository is built upon the following repositories:

- Fair PCA: https://github.com/samirasamadi/Fair-PCA
- LAFTR: https://github.com/VectorInstitute/laftr
- Other code samples: https://github.com/Xianwen-He/Fair_Bench
