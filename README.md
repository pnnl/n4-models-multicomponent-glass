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

An updated citation will be included when available.

Curry, C. E.; Diaz-Acevedo, M; Wang, D; Allec, S. I.; Neway, J. J.; Vienna, J. D.; Lu, X. Boron Coordination in multicomponent glasses: analytical models and machine learning with uncertainty. *Pacific Northwest National Laboratory* **2026**, [Submitted for publication].

```
@unpublished{Curry_Diaz Acevedo_Wang_Allec_Neeway_Vienna_Lu_2026, 
place={Pacific Northwest National Laboratory}, 
author={Curry, Chloe E and Diaz Acevedo, Mayra and 
Wang, Dewei and Allec, Sarah I and Neeway, James J and Vienna, John D and Lu, Xiaonan}, year={2026}}
```
