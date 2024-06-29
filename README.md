![image](https://github.com/DhakshanaMoorthyRDM/LightGBM_Gold_Forecast/assets/121345776/8b09863e-64af-4675-95c2-43e77e47a9d8)

# Gold Price Prediction using Time Series Split and LightGBM

This project focuses on predicting gold prices using time series split and the LightGBM algorithm. The dataset used in this project is sourced from Kaggle, and the goal is to build an accurate predictive model for gold prices.

## Folder Structure

```bash
gold-price-prediction/
│
├── price_prediction.ipynb
└── dataset/
    └── <dataset files>
```

## Project Overview

The objective of this project is to forecast gold prices using advanced machine learning techniques. This is achieved through the following steps:

1.	Import Libraries: Load necessary libraries for data manipulation, visualization, and model building.

2.	Time Series Split: Implement time series split to train and validate the model on sequential data.

3.	Model Building: Utilize the LightGBM regressor for building the predictive model.

4.	Cross-Validation: Perform cross-validation using Repeated K-Fold to evaluate model performance.

5.	Prediction and Evaluation: Predict gold prices and evaluate the model's accuracy using appropriate metrics.

## Libraries Used

This project utilizes the following libraries:

•	pandas: For data manipulation and analysis

•	numpy: For numerical computations

•	matplotlib.pyplot: For plotting and data visualization

•	sklearn.model_selection.TimeSeriesSplit: For time series splitting of data

•	lightgbm.LGBMRegressor: LightGBM model for regression tasks

•	sklearn.model_selection.cross_val_score: For cross-validation scoring

•	sklearn.model_selection.RepeatedKFold: Repeated K-Fold cross-validator

## Getting Started

To get started with the project, follow these steps:

1.	Clone the repository:
```bash
git clone https://github.com/yourusername/gold-price-prediction.git
cd gold-price-prediction
```

2.	Install the required libraries:
```bash
pip install -r requirements.txt

```
3.	Access the dataset from Kaggle:






[kaggle_link 🔗](https://www.kaggle.com/code/dhakshanamoorthyr/lightgbm-gold-forecast)


4.	Open and run the Jupyter Notebook price_prediction.ipynb to explore and execute the project.

## Project Workflow

1.	Import Libraries: Import all necessary libraries for data handling and modeling.

```bash
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit
from lightgbm import LGBMRegressor
from sklearn.model_selection import cross_val_score, RepeatedKFold
```

2.	Time Series Split: Implement time series splitting to handle sequential data for training and validation.

3.	Model Building: Use the LightGBM regressor to build a predictive model for gold prices.

4.	Cross-Validation: Evaluate the model's performance using cross-validation techniques such as Repeated K-Fold.

5.	Prediction and Evaluation: Make predictions using the trained model and assess its accuracy using appropriate metrics.

## Acknowledgements
•	The developers and maintainers of the libraries used in this project.
•	Kaggle for providing the gold price dataset.

Feel free to explore, contribute, and provide feedback!




