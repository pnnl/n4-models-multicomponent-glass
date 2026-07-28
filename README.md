# Boron coordination in multicomponent borate and borosilicate glasses: analytical models and machine learning with uncertainty #

Prediction of boron coordination may provide insight into the relationship between glass composition, structure, and properties, allowing glasses to be designed around desired properties. This package provides trained machine learning (ML) and analytical models for prediction of N<sub>4</sub>, as well as the code used to train the models. Modified Bernstein, modified Du Stebbins, heteroskedastic DNN, PBNN, and GPR models were trained on a diverse multicomponent glass dataset to predict N<sub>4</sub> values. The ML models achieved R<sup>2</sup> values of 0.91 on an isolated test dataset and analytical models achieved R<sup>2</sup> of ~ 0.77. The ML models made available in this work are unique in their ability to provide prediction uncertainty values alongside their N<sub>4</sub> predictions, as well as the inclusion of cooling rate as an input parameter. 

## How to use
To predict boron coordination on your dataset, you can use the `n4_models_predict.ipynb` file in the `Predict_N4` folder. In that notebook, you will be walked through the steps to format your data for prediction on our models. Required columns for prediction are: composition (in mol fraction) and cooling rate (one of Slow cooled, Air quench, Water quench, or Fast quench). An input file template can also be found at `./data/input_format_template.xlsx`.

If you would like to predict only the analytical models, the `N4 Analtical Models Calculation.xlsx` provides an excel file where you can supply one glass composition and see the predicted N<sub>4</sub> amount for the modified Du Stebbins and Bernstein models.

To view the ML model training code, you can explore the files in `ML_Models`.

## Data
The data used in this work can be found in the `data` folder. `boron_coord_final.xlsx` contains the data used after standardization and outlier removal. 

## Setting up the environment
To clone this repository, use `git clone https://github.com/pnnl/n4-models-multicomponent-glass.git`

It is recommend to use [`conda`](https://docs.conda.io/projects/conda/en/latest/index.html) to manage `python` environments. Once `conda` is installed, create a new conda environment using:

```
conda create -n myenv python=3.10
```

Feel free to replace `myenv` with a more unique name for this work (e.g., I call mine `n4-models`). Next, activate the environment:

```
conda activate myenv
```

Finally, run

```
pip install -r requirements.txt
```

## Citation

C. E. Curry, M. Diaz-Acevedo, D. Wang, et al. “Boron Coordination in Multicomponent Glasses: Analytical Models and Machine Learning With Uncertainty.” *International Journal of Applied Glass Science* **17**, no. 3 (2026): e70043. https://doi.org/10.1111/ijag.70043

```
@article{https://doi.org/10.1111/ijag.70043,
author = {Curry, Chloe E. and Diaz-Acevedo, Mayra and Wang, Dewei and Allec, Sarah I. and Neeway, James J. and Vienna, John D. and Lu, Xiaonan},
title = {Boron Coordination in Multicomponent Glasses: Analytical Models and Machine Learning With Uncertainty},
journal = {International Journal of Applied Glass Science},
volume = {17},
number = {3},
pages = {e70043},
keywords = {boron coordination, glass structure, machine learning},
doi = {https://doi.org/10.1111/ijag.70043},
url = {https://ceramics.onlinelibrary.wiley.com/doi/abs/10.1111/ijag.70043},
eprint = {https://ceramics.onlinelibrary.wiley.com/doi/pdf/10.1111/ijag.70043},
abstract = {ABSTRACT Borosilicate glasses are extensively used in a variety of applications from kitchenware to nuclear waste immobilization due to the strong network formed by the Si─O─B bond that makes it resistant to chemical corrosion and gives it a low thermal expansion. Boron, however, exists in both trigonal BO3 and tetrahedral BO4 bonds in glass systems, which impacts the chemical durability and thermal resistance of the glass, among other properties. The fraction of four-coordinated boron (N4 = BO4/[BO3 + BO4]) within a glass may aid in predicting these properties, but it is difficult to derive without experimental data due to the complexity of impacts from varied glass compositions and processing factors. For this reason, compositional models have been developed to predict N4, but the models typically include a limited number (< 5) of components. To help fill this gap in the models, in this work, a diverse multicomponent glass dataset of 809 glasses is compiled from a literature search, and then a number of analytical and machine learning (ML) models are trained on the dataset. Previously developed modified Bernstein and modified Du and Stebbins (MDS) analytical models were fitted to update parameters with the new dataset. Then, partially Bayesian neural networks, Gaussian process regression, and heteroskedastic deterministic neural networks were evaluated. The ML models examined all have different strategies to overcome the potential for overfitting as a result of a limited training dataset, and return results that account for model uncertainty, which can be valuable for understanding model reliability. For the first time, cooling rate is introduced as an input parameter for ML models, showing consistent improvements in performance and solidifying the importance of including parameters outside of composition alone for N4 prediction. The ML models examined here show promise in accurate predictions of N4 in borosilicate glasses, all achieving R2 values of 0.91. These accurate predictions can be used to inform glass design around specific structural properties, accelerating the speed of scientific development.},
year = {2026}
}

```
